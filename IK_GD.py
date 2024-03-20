import numpy as np
from numpy import pi, cos, sin, arctan2
import dill

class IK_GD(object):
	def __init__(self):
		self.f_new = dill.load(open("Jacobian", "rb"))

	def run(self) -> np.ndarray:
		angles = [0, 0, 0, 0, 0, 0]
		p, r = self.demo(0)
		out, angles = self._compute(angles, p, r)

		for i in range(1, 3):
			print(i)
			p, r = self.demo(i)
			a, angles = self._compute(angles, p, r)
			out = np.r_[out, a]

		# print(out)
 		
	def demo(self, target: int) -> list:
		match target:
			case 0:
				p1 = [0.0, -0.194, 0.69]
				r1 = [-90, 0, -180]
				return p1, r1

			case 1:
				p2 = [-0.21, -0.24, 0.2]
				r2 = [-180, 0, -180]
				return p2, r2

			case 2:
				p3 = [-0.21,-0.24, 0.326]
				r3 = [-180, 0, -180]
				return p3, r3

			# case 3:
			# 	p4 = [-0.21,-0.24, 0.326]
			# 	r4 = [-29, 52, -140]
			# 	return p4, r4

			# case 4:
			# 	p4 = [-0.21,-0.413, 0.326]
			# 	r4 = [-29, 52, -140]
			# 	return p4, r4
            
			case _:
				raise ("No exist target")

	def _comp_trans_mat_target(self, target: np.ndarray) -> np.ndarray:
		"""
		Computation target Forward Kinematics D-H table

		:param target: translation, rotation of target
		:returns: 4x4 transformation matrix
		"""
		x = target[0]
		y = target[1]
		z = target[2]
		
		alfa = target[3]
		beta = target[4]
		gamma = target[5]
		
		R_x = np.array([
			[1, 0,           0        ],
			[0, cos(alfa),  -sin(alfa)],
			[0, sin(alfa),   cos(alfa)] 
		])
		
		R_y = np.array([
			[cos(beta),  0, sin(beta)],
			[0,          1, 0        ],
			[-sin(beta), 0, cos(beta)] 
		])
		
		R_z = np.array([
			[cos(gamma), -sin(gamma),  0], 
			[sin(gamma),  cos(gamma),  0],
			[0,           0,           1]
		])

		R = R_z @ R_y @ R_x

		return np.r_[
			np.c_[
				R, np.array([x,y,z])
			], [np.array([0,0,0,1])]
		]
	
	def _get_jacob_mat(self, th1, th2, th3, th4, th5, th6) -> np.ndarray:
		return self.f_new(th1, th2, th3, th4, th5, th6).astype('float64')	

	def _fwd_kinematic(self, th: np.ndarray) -> np.ndarray:
		"""
		Computation of Forward Kinematics by classics D-H tables  

		:param th: joints angle
		:returns: 4x4 transformation matrix
		"""
		# D-H parameters of UR3
		o = [pi/2, 0, 0, pi/2, -pi/2, 0] # Link twist
		d = [0.1519, 0, 0, 0.11235, 0.08535, 0.0819] # Link Offset
		a = [0, -0.24365, -0.21325, 0, 0, 0] # Link Length
		
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

	def _compute(self, angles: list, target_pos: list, target_rot: list) -> np.ndarray:
		angles = np.deg2rad(angles)
		target_rot = np.deg2rad(target_rot)
		target = np.append(target_pos, target_rot)
  
		i = 0
		q = np.array([angles])
  
		trans_mat_target = self._comp_trans_mat_target(target)
		trans_mat_current = self._fwd_kinematic(q[i,:])
		print("init: ", trans_mat_current)
		print("targ: ", trans_mat_target)
  
		# error computation
		error = np.linalg.norm(trans_mat_current - trans_mat_target)	
		error = np.linalg.norm(trans_mat_current[0,3] - trans_mat_target[0,3])	
		print("error: ", error)
  
		while error > 0.001:
			jacob_mat = self._get_jacob_mat(q[i,0],q[i,1],q[i,2],q[i,3],q[i,4],q[i,5])
			sk = 1
			min_err = np.inf
			q_min = np.array([])
			for j in range(6):
				tmp = q[i,:] - jacob_mat[j,:] * sk
				# compute new error
				trans_mat_current = self._fwd_kinematic(tmp)
				error = np.linalg.norm(trans_mat_current - trans_mat_target)
				error = np.linalg.norm(trans_mat_current[0,3] - trans_mat_target[0,3])

				if error < min_err:
					q_min = tmp
					min_err = error

			q = np.r_[q, [q_min]]

				# limit computed joint from 360° to 180° -> prevent some self collision 
				# if(np.any(q[-1,:] > pi) or np.any(q[i+1,:] < -pi)): 
				# 	l = np.argwhere(q[-1,:] > pi)
				# 	k = np.argwhere(q[-1,:] < -pi)
				# 	q[-1,l] = q[i,l]
				# 	q[-1,k] = q[i,k]

			i += 1
			print(f"iter= {i}, error= {error}")

		goal = q[-1,:]

		# generate simple path, with 100 samples
		nr_pnts = 100
		a = np.zeros((nr_pnts, 6))

		for i in range(5):
			a[:,i] = np.linspace(np.rad2deg(angles[i]), np.rad2deg(goal[i]), nr_pnts)

		return a, np.rad2deg(goal)
  
IK_solver = IK_GD()
IK_solver.run()