import numpy as np
from numpy import pi, cos, sin, arctan2
import dill

class IK_GD(object):
	def __init__(self):
		# with open("Jacob.pkl", "wb") as d:
		# 	tmp = dill.load(open("Jacobian", "rb"))
		# 	dill.dump(tmp, d)

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

		print(out)
 		
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

			case 3:
				p4 = [-0.21,-0.24, 0.326]
				r4 = [-29, 52, -140]
				return p4, r4

			case 4:
				p4 = [-0.21,-0.413, 0.326]
				r4 = [-29, 52, -140]
				return p4, r4
            
			case _:
				raise ("No exist target")

	def _get_target_trans_mat(self, target: np.ndarray) -> np.ndarray:
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

	def _get_orient(self, T_desired: np.ndarray, T_current: np.ndarray) -> np.ndarray:
		"""
		Computation angle-axis distance

		:param T_desired: d-h table of target
		:param T_current: d-h table current state
		:returns: 6x1 array translation, rotation  
		"""
		Td = T_desired[:3,3]
		Ti = T_current[:3,3]

		Rd = T_desired[:3,:3]
		Ri = T_current[:3,:3]

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
		
		return np.r_[np.array([Td-Ti]).T, a]
 
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

		step_size_descent_rate = 0.999
		step_size = 1.2
  
		while error > 0.001:
      
			jacob_mat = self._get_jacob_mat(q[i,0],q[i,1],q[i,2],q[i,3],q[i,4],q[i,5])

			orient = self._get_orient(trans_mat_target, trans_mat_current)

			step_size = step_size * step_size_descent_rate

			tmp = q[-1,:] + (np.linalg.pinv(jacob_mat) @ orient).T[0,:] * step_size

			# compute new error
			trans_mat_current = self._fwd_kinematic(tmp)
			
			error = np.linalg.norm(trans_mat_current - trans_mat_target)

			q = np.r_[q, [tmp]]

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