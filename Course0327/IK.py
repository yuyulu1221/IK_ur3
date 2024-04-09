#%% import module
import numpy as np
from numpy import pi, cos, sin, arctan2
from Jacob import Jacob

#%% IK class
class IK(object):
	def __init__(self, angles, pos, rot):
		self.Jacob = Jacob()
		self.angles = np.deg2rad(angles)
		self.pos = pos
		self.rot = rot

	def get_target_trans_mat(self, target: np.ndarray) -> np.ndarray:
	
 		# mm -> m
		target[:3] /= 1000

		# Get rotation matrix by angle axis
		R = self.axis_angle_2_rot_mat(target[3:6])

		# Combine to transition matrix
		return np.r_[
			np.c_[
				R, np.array(target[:3].T)
			], [np.array([0,0,0,1])]
		]
	
	def get_jacob_mat(self, th1, th2, th3, th4, th5, th6) -> np.ndarray:
		
		return self.Jacob.get_jacob(th1, th2, th3, th4, th5, th6)

	def fwd_kinematic(self, th: np.ndarray) -> np.ndarray:
		"""
		Computation of Forward Kinematics by classics D-H tables  

		:param th: joints angle
		:returns: 4x4 transformation matrix
		"""
		# D-H parameters of UR3e
		o = [pi/2, 0, 0, pi/2, -pi/2, 0] # Link twist
		d = [0.15185, 0, 0, 0.13105, 0.08535, 0.0921] # Link Offset
		a = [0, -0.24355, -0.2132, 0, 0, 0] # Link Length
		
		# Using D-H table to generate transformation matrices
		for i in range(6):
			A_x = np.array([
				[cos(th[i]),  -sin(th[i]) * cos(o[i]),     sin(th[i]) * sin(o[i]),     a[i] * cos(th[i])], 
				[sin(th[i]),   cos(th[i]) * cos(o[i]),    -cos(th[i]) * sin(o[i]),     a[i] * sin(th[i])],
				[0       ,     sin(o[i]),                  cos(o[i]),                  d[i]             ],
				[0       ,     0,                          0,                          1                ]
			])

			if i == 0:
				A = A_x
			else:
				A = A @ A_x 
		
		return A

	def axis_angle_2_rot_mat(self, axis_angle):
		
		th = np.linalg.norm(axis_angle)
  
		# normalize
		axis_angle /= th		

		Rx = axis_angle[0]
		Ry = axis_angle[1]
		Rz = axis_angle[2]
  
		R_mat = np.array([
			[cos(th)+Rx**2*(1-cos(th)), 	Rx*Ry*(1-cos(th))-Rz*sin(th), 	Rx*Rz*(1-cos(th))+Ry*sin(th)],
			[Ry*Rx*(1-cos(th))+Rz*sin(th),	cos(th)+Ry**2*(1-cos(th)),		Ry*Rz*(1-cos(th))-Rx*sin(th)],
			[Rz*Rx*(1-cos(th))-Ry*sin(th),	Rz*Ry*(1-cos(th))+Rx*sin(th),	cos(th)+Rz**2*(1-cos(th))]
		])

		return R_mat

	def rot_mat_2_axis_angle(self, Rd, Ri) -> np.ndarray:
		
		R = Rd @ Ri.T
		l = np.array([ 
			[R[2,1] - R[1,2]],
			[R[0,2] - R[2,0]],
			[R[1,0] - R[0,1]]
		])
		
		l_length = np.linalg.norm(l)

		if(l_length > 0):
			a = ((arctan2(l_length, R[0,0] + R[1,1] + R[2,2] - 1 ) ) / l_length) * l

		else:  
			if(R[0,0] + R[1,1] + R[2,2] > 0):
				a = np.array([[0,0,0]]).T
			
			else:
				a = pi/2 * np.array([[
					R[0,0] + 1,
					R[1,1] + 1,
					R[2,2] + 1
				]]).T
    
		return a

	def get_orient(self, T_desired: np.ndarray, T_current: np.ndarray) -> np.ndarray:
		"""
		:param T_desired: d-h table of target
		:param T_current: d-h table current state
		:returns: 6x1 array translation + rotation  
		"""
		Td = T_desired[:3,3]
		Ti = T_current[:3,3]

		Rd = T_desired[:3,:3]
		Ri = T_current[:3,:3]

		R = self.rot_mat_2_axis_angle(Rd, Ri)
		
		return np.r_[np.array([Td-Ti]).T, R]	
 
	def compute(self, angles: list, target_pos: list, target_rot: list) -> np.ndarray:
		
		angles = np.deg2rad(angles)
		target = np.append(target_pos, target_rot)
 
		i = 0
		q = np.array([angles])
  
		trans_mat_target = self.get_target_trans_mat(target)
		trans_mat_current = self.fwd_kinematic(q[i,:])
  
		# error computation
		error = np.linalg.norm(trans_mat_current - trans_mat_target)
  	
		# step_size_descent_rate = 0.999
		step_size = 0.8
  
		while error > 0.002:
			jacob_mat = self.get_jacob_mat(q[i,0],q[i,1],q[i,2],q[i,3],q[i,4],q[i,5])

			# compute the difference between target and current state
			orient = self.get_orient(trans_mat_target, trans_mat_current)

			tmp = q[-1,:] + (np.linalg.pinv(jacob_mat) @ orient).T[0,:] * step_size

			# compute new error
			trans_mat_current = self.fwd_kinematic(tmp)
			
			# compute the error with transformation matrix
			error = np.linalg.norm(trans_mat_current - trans_mat_target)
   
			q = np.r_[q, [tmp]]
   
			# limit computed joint from 360° to 180° -> prevent some self collision 
			if(np.any(q[-1,:] > pi) or np.any(q[i+1,:] < -pi)): 
				l = np.argwhere(q[-1,:] > pi)
				k = np.argwhere(q[-1,:] < -pi)
				q[-1,l] = q[i,l]
				q[-1,k] = q[i,k]

			i += 1
			print(f"it={i})	error= {error}")

		goal = q[-1,:]

		return np.rad2deg(goal)

	def run(self) -> np.ndarray:
		print(self.compute(self.angles, self.pos, self.rot).round(2))

#%% main
np.set_printoptions(suppress=True)
init_joint_angles = [-110, -110, -75, -90, 130, 0]
tcp_pos = [-118.39, -376.08, 195.88]
tcp_rot = [0.488, 1.987, 0.438]

IK_solver = IK(init_joint_angles, tcp_pos, tcp_rot)
IK_solver.run()