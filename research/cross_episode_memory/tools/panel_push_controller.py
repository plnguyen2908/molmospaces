"""Actuator-only panel pushing after a partial handle pull.

Approaches use cuRobo; contact arcs use local IK checked against MuJoCo meshes.
The contact exemption is limited to the selected panel and right hand bodies.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from research.cross_episode_memory.door_contact import (
    HAND_BODIES, contact_path_collision, external_door_contacts,
    hand_panel_contact, panel_pose, solve_contact_ik,
)
from research.cross_episode_memory.tools.check_fridge_transfer import F, NS


class PanelPushController:
    def allowed_panel_contact(self, robot, other):
        return (getattr(self, 'panel_push_active', False)
                and robot in HAND_BODIES and other == self.model.jnt_bodyid[self.jid])

    def before_step(self):
        super().before_step()
        if not getattr(self, 'panel_push_active', False):
            return
        contact = hand_panel_contact(self.model, self.data, self.model.jnt_bodyid[self.jid])
        metrics = self.report.setdefault('panel_push_contact', dict(
            maximum_force_n=0., maximum_depth_m=0., force_contact_steps=0))
        metrics['maximum_force_n'] = max(metrics['maximum_force_n'], contact['normal_force_n'])
        metrics['maximum_depth_m'] = max(metrics['maximum_depth_m'], contact['depth_m'])
        metrics['force_contact_steps'] += int(contact['normal_force_n'] > .1)
        if contact['depth_m'] > .001:
            raise RuntimeError('Hand/panel penetration exceeded 1 mm during push')
        if contact['normal_force_n'] > 60.:
            raise RuntimeError('Hand/panel push force exceeded 60 N')
        if self.unintended_penetration() > .003:
            bad = contact_path_collision(self.model,self.data,self.handle_bid,int(self.model.jnt_bodyid[self.jid]))
            self.report['rejected_panel_pose'] = dict(qpos=self.data.qpos.tolist(),contact=bad)
            raise RuntimeError(f'Robot/scenery collision during panel push: {bad}')
        bad = external_door_contacts(self.model, self.data, self.moving_door_bodies(), F)
        if any(c['depth_m'] > .001 for c in bad):
            raise RuntimeError(f'Moving door collided with scenery: {bad}')

    def moving_door_bodies(self):
        root = int(self.model.jnt_bodyid[self.jid])
        bodies = {root}
        for bid in range(root+1, self.model.nbody):
            if self.model.body_parentid[bid] in bodies:
                bodies.add(bid)
        return bodies

    def panel_servo(self, pose, expected_angle, seconds=.12):
        """Validate a short Cartesian segment before sending joint controls."""
        start = self.tcp()
        angle = self.angle()
        addresses = [self.model.jnt_qposadr[self.model.joint(NS+n).id] for n in self.planner.names]
        q = self.data.qpos[addresses].copy()
        count = max(2, int(np.ceil(np.linalg.norm(pose[:3,3]-start[:3,3])/.003)))
        rotations = Slerp([0,1], Rotation.from_matrix([start[:3,:3], pose[:3,:3]]))
        probe = mujoco.MjData(self.model)
        path = []
        for f in np.linspace(0,1,count+1)[1:]:
            target = start.copy()
            target[:3,3] = (1-f)*start[:3,3]+f*pose[:3,3]
            target[:3,:3] = rotations(f).as_matrix()
            probe.qpos[:] = self.data.qpos
            probe.qpos[self.model.jnt_qposadr[self.jid]] = (1-f)*angle+f*expected_angle
            allowed_panel = int(self.model.jnt_bodyid[self.jid]) if getattr(self,'panel_push_active',False) else None
            handle_id = -1 if getattr(self,'released_handle_motion',False) else self.handle_bid
            q = solve_contact_ik(self.model,probe,target,self.planner.names,q,handle_id,allowed_panel)
            probe.qpos[addresses] = q
            mujoco.mj_forward(self.model,probe)
            bad = contact_path_collision(self.model,probe,handle_id,allowed_panel)
            if bad:
                raise RuntimeError(f'Unsafe panel contact path: {bad}')
            external = external_door_contacts(self.model,probe,self.moving_door_bodies(),F)
            if any(c['depth_m'] > .0005 for c in external):
                raise RuntimeError(f'Panel path blocked by scenery: {external}')
            path.append(q.copy())
        for q in path:
            self.data.ctrl[self.arm_aids] = self.bounded_arm_command(q)
            self.tick(seconds/count)
            if self.unintended_penetration() > .003:
                raise RuntimeError('Unexpected contact during door hand movement')
        self.tick(.15)
        error=float(np.linalg.norm(self.tcp()[:3,3]-pose[:3,3]))
        if error>.006:
            raise RuntimeError(f'Door hand segment missed target by {error:.4f} m at {self.stage}')
        if not getattr(self,'panel_push_active',False) and abs(self.angle()-angle)>np.radians(1.):
            raise RuntimeError('Door moved during a planned non-contact hand segment')

    def mesh_free_move(self, stage, pose):
        """Reject cuRobo paths that intersect actual meshes; refine the goal if needed."""
        self.stage=stage
        names=self.planner.names
        addresses=[self.model.jnt_qposadr[self.model.joint(NS+n).id] for n in names]
        limits=self.model.jnt_range[[self.model.joint(NS+n).id for n in names]]
        start=self.data.qpos[addresses].copy()
        probe=mujoco.MjData(self.model)
        rejected=[]
        def checked(path):
            previous=start
            for q in path:
                if np.any(q<limits[:,0]-1e-5) or np.any(q>limits[:,1]+1e-5):
                    rejected.append({'reason':'planner joint limit violation'});return False
                count=max(1,int(np.ceil(np.max(abs(q-previous))/.005)))
                for f in np.linspace(0.,1.,count+1)[1:]:
                    probe.qpos[:]=self.data.qpos;probe.qpos[addresses]=previous+f*(q-previous)
                    mujoco.mj_forward(self.model,probe)
                    bad=contact_path_collision(self.model,probe,-1)
                    if bad:
                        rejected.append(bad);return False
                previous=q
            return True
        def direct(target):
            count=max(2,int(np.ceil(1.9*np.max(abs(target-start))/.005)))
            u=np.linspace(0.,1.,count+1)[1:];f=10*u**3-15*u**4+6*u**5
            return start[None,:]+f[:,None]*(target-start)[None,:]
        self.load_world()
        goal=list(pose[:3,3]-[0,0,.005])+Rotation.from_matrix(pose[:3,:3]).as_quat(scalar_first=True).tolist()
        selected=None;seeds=[];method='cuRobo with actual-mesh validation';dt=self.planner.dt*self.args.motion_slowdown
        try:
            candidate=self.planner.plan(start.tolist(),goal)
            seeds.append(np.clip(candidate[-1],limits[:,0]+.04,limits[:,1]-.04))
            if checked(candidate):selected=candidate
        except RuntimeError as exc:
            rejected.append({'planner_error':str(exc)})
        if selected is None:
            wrist_sign=1. if self.active_door=='left' else -1.
            preferred=dict(zip((f'right_arm_{i}' for i in range(7)),(-.8,-.4,-.6,-1.2,-.6,.6*wrist_sign,-.4*wrist_sign)))
            seeds.append(np.array([preferred.get(n,0.) for n in names]))
            for seed in seeds:
                probe.qpos[:]=self.data.qpos
                try:
                    target=solve_contact_ik(self.model,probe,pose,names,seed,-1,
                                            trust_radius=10.,joint_margin=.04,max_nfev=200)
                except RuntimeError as exc:
                    rejected.append({'physical_ik_error':str(exc)});continue
                # Keep cuRobo for the free-space trajectory when its path passes
                # the same actual-mesh check used by physical execution.
                try:
                    candidate=self.planner.plan_joints(start.tolist(),target.tolist())
                    if checked(candidate):selected=candidate;break
                except RuntimeError as exc:
                    rejected.append({'planner_error':str(exc)})
                candidate=direct(target)
                if checked(candidate):
                    selected=candidate;method='collision-constrained IK and mesh-checked joint path';dt=.02;break
        self.report.setdefault('door_free_motion_plans',[]).append(dict(stage=stage,rejected=rejected,accepted=selected is not None))
        if selected is None:
            raise RuntimeError(f'No actual-mesh-clear door approach: {stage}; {rejected}')
        bias=np.zeros(len(names))
        for q in selected:
            self.data.ctrl[self.arm_aids]=self.bounded_arm_command(q+bias)
            self.tick(dt)
            if self.unintended_penetration()>.003:
                raise RuntimeError('Unexpected contact during validated door approach')
            bias=np.clip(self.data.ctrl[self.arm_aids]-q+.3*(q-self.data.qpos[addresses]),-.08,.08)
        target=np.asarray(selected[-1])
        for _ in range(50):
            self.data.ctrl[self.arm_aids]=self.bounded_arm_command(target+bias);self.tick(.02)
            bias=np.clip(self.data.ctrl[self.arm_aids]-target+.3*(target-self.data.qpos[addresses]),-.08,.08)
        error=float(np.linalg.norm(self.tcp()[:3,3]-pose[:3,3]))
        self.record(door_approach_method=method,tcp_error_m=error,rejected_plans=len(rejected))
        if error>.02:
            raise RuntimeError(f'Validated door approach tracking error: {error:.4f} m')

    def push_panel_arc(self, target_angle, approach=True):
        """Establish hand contact and push; never write a live hinge position."""
        opening = abs(target_angle) > abs(self.angle())
        label = 'opening' if opening else 'closing'
        self.panel_push_active = not approach
        was_released=getattr(self,"released_handle_motion",False)
        self.released_handle_motion=True
        try:
            pose, direction, radial = panel_pose(self.model,self.data,self.jid,
                                                  self.door_open_sign,opening,**getattr(self,"panel_contact_profile",{}))
            self.data.actuator(NS+'right_finger_act').ctrl[0] = 0.
            self.tick(.4)
            if approach:
                pre = pose.copy(); pre[:3,3] -= direction*.025
                edge = pre.copy()
                edge[:3,3] = self.data.xanchor[self.jid]+.70*radial-.10*direction
                edge[2,3] = pre[2,3]
                self.mesh_free_move('approach free door edge',edge)
                self.stage = 'reach '+('interior' if opening else 'exterior')+' panel from free edge'
                self.panel_servo(pre,self.angle(),seconds=max(.8,float(np.linalg.norm(pre[:3,3]-edge[:3,3]))/.08))
                self.panel_push_active = True
                self.stage = 'contact panel for '+label
                self.panel_servo(pose,self.angle(),seconds=.8)
            else:
                self.stage = 'contact panel for '+label
            initial_angle = self.angle()
            contact_start = self.report.get('panel_push_contact',{}).get('force_contact_steps',0)
            search_origin = self.tcp()
            for search_step in range(35):
                contact = hand_panel_contact(self.model,self.data,self.model.jnt_bodyid[self.jid])
                if contact['normal_force_n'] > .2 or abs(self.angle()-initial_angle) > np.radians(.2):
                    break
                pose = search_origin.copy();pose[:3,3] += direction*.001*(search_step+1)
                self.panel_servo(pose,self.angle(),seconds=.12)
            else:
                raise RuntimeError('Hand did not establish physical panel contact')
            initial_angle = self.angle(); start = self.tcp()
            anchor = self.data.xanchor[self.jid].copy();axis = self.data.xaxis[self.jid].copy()
            # Keep the initial measured hand/panel relationship along the arc.
            self.stage = 'push panel '+label
            steps = max(1,int(np.ceil(abs(target_angle-initial_angle)/np.radians(.5))))
            worst_tracking = 0.
            for angle in np.linspace(initial_angle,target_angle,steps+1)[1:]:
                rot = Rotation.from_rotvec(axis*(angle-initial_angle)).as_matrix()
                goal = start.copy();goal[:3,:3] = rot@start[:3,:3]
                goal[:3,3] = anchor+rot@(start[:3,3]-anchor)
                self.panel_servo(goal,float(angle),seconds=.18)
                error = abs(self.angle()-angle)
                worst_tracking = max(worst_tracking,error)
                if error > np.radians(3.):
                    raise RuntimeError(f'Panel stopped following the hand during {label}: {np.degrees(error):.2f} deg')
            self.tick(.3)
            contact_steps = self.report['panel_push_contact']['force_contact_steps']-contact_start
            if contact_steps < 10:
                raise RuntimeError('No sustained force evidence for panel motion')
            self.record(panel_action=label,side=self.active_door,door_angle_deg=abs(np.degrees(self.angle())),
                        force_contact_steps=contact_steps,maximum_tracking_error_deg=np.degrees(worst_tracking))
            # Unload the contact before moving along the panel's free edge.
            # Sliding while still pressing can catch a finger on the door lip.
            rotation = Rotation.from_rotvec(axis*(self.angle()-initial_angle))
            direction = rotation.apply(direction)
            radial = rotation.apply(radial)
            self.stage = 'unload hand after panel '+label
            pose = self.tcp();pose[:3,3] -= .03*direction
            self.panel_servo(pose,self.angle(),seconds=.8)
            self.stage = 'withdraw hand past panel edge after '+label
            pose = self.tcp();pose[:3,3] += .15*radial
            self.panel_servo(pose,self.angle(),seconds=1.5)
        finally:
            self.panel_push_active = False
            self.released_handle_motion=was_released

    def tuck_arm(self, seconds=2.0):
        if (getattr(self.args,'panel_push_doors',False) and not self.attached
                and not getattr(self,'holding_loaf',False)):
            targets={f'right_arm_{i}':(-.02 if i==3 else 0.) for i in range(7)}
            targets.update({f'torso_{i}':0. for i in range(6)})
            if all(abs(float(self.data.joint(NS+n).qpos[0])-q)<.01 for n,q in targets.items()):
                self.stage='tuck arm';self.record(tucked=True,tuck_method='already at travel posture')
                return
        if not getattr(self.args,'panel_push_doors',False) or self.attached or getattr(self,'holding_loaf',False):
            return super().tuck_arm(seconds)
        self.stage='tuck arm'
        self.planner=self.make_planner();self.arm_aids=self.actuator_ids(self.planner.names)
        self.load_world()
        names=self.planner.names
        addresses=[self.model.jnt_qposadr[self.model.joint(NS+n).id] for n in names]
        current=self.data.qpos[addresses].copy()
        travel=np.array([-.02 if n=='right_arm_3' else 0. for n in names])
        probe=mujoco.MjData(self.model)
        candidate=None;planner_error=None;selected=None;rejected=[];dt=.02
        try:
            candidate=self.planner.plan_joints(current.tolist(),travel.tolist())
        except RuntimeError as exc:
            planner_error=str(exc)
        if candidate is not None:
            previous=current;bad=None
            for q in candidate:
                for f in np.linspace(0.,1.,max(2,int(np.ceil(np.max(abs(q-previous))/.005))+1))[1:]:
                    probe.qpos[:]=self.data.qpos;probe.qpos[addresses]=previous+f*(q-previous)
                    mujoco.mj_forward(self.model,probe)
                    bad=contact_path_collision(self.model,probe,-1)
                    if bad:break
                if bad:break
                previous=q
            if bad:rejected.append(bad)
            else:selected=candidate;dt=self.planner.dt*self.args.motion_slowdown
        # A conservative planner can reject a valid start near a released door.
        # Every fallback also checks the entire path against the actual meshes.
        folded=dict(zip((f'right_arm_{i}' for i in range(7)),(.5,0.,0.,-2.3,0.,-.5,0.)))
        fold_here=np.array([folded.get(n,current[i]) for i,n in enumerate(names)])
        fold_upright=np.array([folded.get(n,0.) for n in names])
        for waypoints in ([] if selected is not None else ([travel],[fold_here,travel],[fold_here,fold_upright,travel])):
            path=[];start=current;bad=None
            for end in waypoints:
                samples=max(2,int(np.ceil(1.9*np.max(abs(end-start))/.005)))
                for u in np.linspace(0.,1.,samples+1)[1:]:
                    f=10*u**3-15*u**4+6*u**5
                    q=start+f*(end-start)
                    probe.qpos[:]=self.data.qpos;probe.qpos[addresses]=q
                    mujoco.mj_forward(self.model,probe)
                    bad=contact_path_collision(self.model,probe,handle_id=-1)
                    if bad:break
                    path.append(q.copy())
                if bad:break
                start=end
            if bad:
                rejected.append(bad);continue
            selected=path;break
        if selected is None:
            raise RuntimeError(f'No mesh-clear empty-arm tuck after cuRobo rejection: {rejected}')
        self.stage='tuck empty arm along mesh-checked path'
        bias=np.clip(self.data.ctrl[self.arm_aids]-current,-.08,.08)
        for q in selected:
            self.data.ctrl[self.arm_aids]=self.bounded_arm_command(q+bias);self.tick(dt)
            if self.unintended_penetration()>.003:
                raise RuntimeError('Physical contact during empty-arm tuck')
            bias=np.clip(self.data.ctrl[self.arm_aids]-q+.5*(q-self.data.qpos[addresses]),-.08,.08)
        self.data.ctrl[self.arm_aids]=self.bounded_arm_command(travel)
        self.tick(1.)
        if np.max(abs(self.data.qpos[addresses]-travel))>.05:
            raise RuntimeError('Empty-arm tuck did not settle at its travel posture')
        self.record(tucked=True,tuck_method='mesh-checked joint path',planner_error=planner_error,
                    path_samples=len(selected),rejected_paths=len(rejected))

    def retreat(self, stage):
        if not getattr(self.args,'panel_push_doors',False):
            return super().retreat(stage)
        # Clear the released handle before changing wrist orientation. The old
        # simultaneous 12 cm retreat and rotation exceeded the left wrist's reach.
        self.stage = stage
        pose = self.tcp();pose[0,3] -= .14
        self.released_handle_motion = True
        try:
            self.panel_servo(pose,self.angle(),seconds=2.2)
            self.tick(.3)
        finally:
            self.released_handle_motion = False
        if self.handle_contacts():
            raise RuntimeError('Handle withdrawal still has finger contact')
        self.record(handle_withdrawal_m=.14,tcp_error_m=float(np.linalg.norm(self.tcp()[:3,3]-pose[:3,3])))

    def finish_panel_opening(self):
        self.tuck_arm()
        side = self.active_door
        stance = (.05,1.95,.5) if side == 'left' else (.05,1.90,-.35)
        self.navigate(np.asarray(stance[:2]),carrying=False,face=stance[2])
        self.planner = self.make_planner()
        self.arm_aids = self.actuator_ids(self.planner.names)
        self.push_panel_arc(self.door_open_sign*np.radians(self.args.open_angle))
        self.door_opening_modes[side] = 'pull_then_push'

    def live_handle_grasp(self, side_on=False):
        closed=self.handle_grasp_pose()
        rotation=Rotation.from_rotvec(self.data.xaxis[self.jid]*self.angle()).as_matrix()
        anchor=self.data.xanchor[self.jid]
        pose=closed.copy();pose[:3,:3]=rotation@closed[:3,:3]
        pose[:3,3]=anchor+rotation@(closed[:3,3]-anchor)
        if side_on:
            # The pad center is 17.5 mm behind the TCP. Preserve that center,
            # but approach along the panel edge so the wrist clears the wall.
            center=pose[:3,3]-.0175*pose[:3,2]
            radial=rotation@np.array([0.,-self.door_open_sign,0.])
            pose[:3,2]=-radial;pose[:3,0]=[0.,0.,-1.]
            pose[:3,1]=np.cross(pose[:3,2],pose[:3,0])
            pose[:3,3]=center+.0175*pose[:3,2]
        return pose

    def close_native_fridge(self):
        if self.door_opening_modes.get(self.active_door) != 'pull_then_push':
            return super().close_native_fridge()
        self.operating_door = True
        self.set_grip(self.args.door_grip_force,self.args.door_grip_kp)
        try:
            self.tuck_arm()
            stance=(.05,2.15,1.2) if self.active_door=='left' else (.05,1.75,-.5)
            self.navigate(np.asarray(stance[:2]),carrying=False,face=stance[2])
            self.planner=self.make_planner();self.arm_aids=self.actuator_ids(self.planner.names)
            self.gripper(True)
            target=self.live_handle_grasp(side_on=True)
            pre=target.copy();pre[:3,3]-=.10*pre[:3,2]
            self.mesh_free_move('approach handle from free edge',pre)
            self.stage='reach handle from free edge';self.panel_servo(target,self.angle(),seconds=1.5)
            self.stage='grasp open handle from side';self.gripper(False)
            self.follow_hinge(self.door_open_sign*np.radians(20.))
            self.stage='release partially closed handle';self.gripper(True)
            self.released_handle_motion=True
            try:
                target=self.tcp();target[:3,3]-=.10*target[:3,2]
                self.panel_servo(target,self.angle(),seconds=2.)
            finally:
                self.released_handle_motion=False
            self.tuck_arm()
            self.navigate(np.array([self.args.door_stance_x,self.args.door_stance_y]),carrying=False,face=0.)
            self.planner=self.make_planner();self.arm_aids=self.actuator_ids(self.planner.names)
            target=self.live_handle_grasp()
            pre=target.copy();pre[:3,3]-=.10*pre[:3,2]
            self.mesh_free_move('approach handle for final close',pre)
            self.stage='reach handle for final close';self.panel_servo(target,self.angle(),seconds=1.5)
            self.stage='grasp handle for final close';self.gripper(False)
            self.follow_hinge(0.)
            self.stage='release closed handle';self.gripper(True)
            self.retreat('withdraw from closed handle');self.tuck_arm();self.tick(.5)
            if abs(np.degrees(self.angle()))>self.args.close_tolerance:
                raise RuntimeError('Fridge did not remain shut after physical handle closure')
            self.record(door_closed_deg=abs(np.degrees(self.angle())),closing_method='side handle grasp, then frontal final close')
        finally:
            self.set_grip(self.args.grip_force,self.args.grip_kp)
            self.operating_door=False
