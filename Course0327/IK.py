#%% import module
import numpy as np
from numpy import pi, cos, sin, arctan2
import dill
from Jacob import Jacob

#%% IK class
class IK(object):
	def __init__(self, angles=[0,0,0,0,0,0], pos=[0.0, -0.24, 0.24], rot=[-180, 0, -180]):
		self.Jacob = Jacob()
		self.angles = np.deg2rad(angles)
		self.pos = pos
		self.rot = rot

	def test(self):
		print(self._fwd_kinematic(self.angles))
		print(self.axis_angle_2_rot_mat(self.rot))

	
	def run(self) -> np.ndarray:
		print(self._compute(self.angles, self.pos, self.rot))

	def _get_target_trans_mat(self, target: np.ndarray) -> np.ndarray:
		x = target[0]
		y = target[1]
		z = target[2]
  
		R = self.axis_angle_2_rot_mat(target[3:6])

		return np.r_[
			np.c_[
				R, np.array([x,y,z])
			], [np.array([0,0,0,1])]
		]
	
	def _get_jacob_mat(self, th1, th2, th3, th4, th5, th6) -> np.ndarray:
		return self.Jacob.get_jacob(th1, th2, th3, th4, th5, th6)

	def _fwd_kinematic(self, th: np.ndarray) -> np.ndarray:
		"""
		Computation of Forward Kinematics by classics D-H tables  

		:param th: joints angle
		:returns: 4x4 transformation matrix
		"""
		# D-H parameters of UR3e
		o = [pi/2, 0, 0, pi/2, -pi/2, 0] # Link twist
		d = [0.1519, 0, 0, 0.13105, 0.08535, 0.0921] # Link Offset
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
		
		return A.round(4)

	def axis_angle_2_rot_mat(self, axis_angle):
		th = np.linalg.norm(axis_angle)
		axis_angle /= th		
		# th = np.rad2deg(th)
		Rx = axis_angle[0]
		Ry = axis_angle[1]
		Rz = axis_angle[2]
  
		R_mat = np.array([
			[cos(th)+Rx**2*(1-cos(th)), 	Rx*Ry*(1-cos(th))-Rz*sin(th), 	Rx*Rz*(1-cos(th))+Ry*sin(th)],
			[Ry*Rx*(1-cos(th))+Rz*sin(th),	cos(th)+Ry**2*(1-cos(th)),		Ry*Rz*(1-cos(th))-Rx*sin(th)],
			[Rz*Rx*(1-cos(th))-Ry*sin(th),	Rz*Ry*(1-cos(th))+Rx*sin(th),	cos(th)+Rz**2*(1-cos(th))]
		])

		return R_mat.round(4)

	def rot_mat_2_axis_angle(self, Rd, Ri) -> np.ndarray:
		
		R = Rd @ Ri.T
		l = np.array([ 
			[R[2,1] - R[1,2]],
			[R[0,2] - R[2,0]],
			[R[1,0] - R[0,1]]
		])
		
		l_length = np.linalg.norm(l)

		if(l_length > 0):
			# print("???")
			a = ((arctan2(l_length, R[0,0] + R[1,1] + R[2,2] - 1 ) ) / l_length) * l
			# a = [
       		# 	[(R[2, 1] - R[1, 2]) / l_length * pi],
          	# 	[(R[0, 2] - R[2, 0]) / l_length * pi],
   			# 	[(R[1, 0] - R[0, 1]) / l_length * pi]
   			# ]

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

	def _get_orient(self, T_desired: np.ndarray, T_current: np.ndarray) -> np.ndarray:
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
 
	def _compute(self, angles: list, target_pos: list, target_rot: list) -> np.ndarray:
		angles = np.deg2rad(angles)
		target_rot = np.deg2rad(target_rot)
		target = np.append(target_pos, target_rot)
  
		i = 0
		q = np.array([angles])
  
		trans_mat_target = self._get_target_trans_mat(target)
		trans_mat_current = self._fwd_kinematic(q[i,:])
  
		# error computation
		error = np.linalg.norm(trans_mat_current - trans_mat_target)	
		# step_size_descent_rate = 0.999
		step_size = 0.8
  
		while error > 0.001:
			jacob_mat = self._get_jacob_mat(q[i,0],q[i,1],q[i,2],q[i,3],q[i,4],q[i,5])

			# compute the difference between target and current state
			orient = self._get_orient(trans_mat_target, trans_mat_current)

			tmp = q[-1,:] + (np.linalg.pinv(jacob_mat) @ orient).T[0,:] * step_size

			# compute new error
			trans_mat_current = self._fwd_kinematic(tmp)
			
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
			print(f"iter= {i}, error= {error}")

		goal = q[-1,:]

		return np.rad2deg(goal)

if __name__ == "__main__":  
	np.set_printoptions(suppress=True)
	# ori_joint_angles = input("Enter the origin joint angles: ").split()
	# ori_joint_angles = list(map(lambda x: float(x), ori_joint_angles))
 
	# tcp_pos = input("Enter the TCP position: ").split()
	# tcp_pos = list(map(lambda x: float(x) / 1000, tcp_pos))
	# tcp_rot = input("Enter the TCP rot: ").split()
	# tcp_rot = list(map(lambda x: np.rad2deg(float(x)), tcp_rot))
	# print(tcp_pos)
	# print(tcp_rot)
	
	# IK_solver = IK(ori_joint_angles, tcp_pos, tcp_rot)
	# IK_solver.run()
	IK_solver = IK(
     	angles=[-91.51, -98.47, -109.73, -63.28, 91.40, 358.42],
		pos = [-136.86, -303.22, 197.79],
		rot = [0.001, -3.166, -0.040]
    )
	IK_solver.test()