#%% import module
import numpy as np
from numpy import pi, cos, sin, arctan2
import dill
from Jacob import Jacob

#%% FK class
class FK(object):
	def __init__(self, th):
		self.th = np.deg2rad(th)
		
	def fwd_kinematic(self, th: np.ndarray) -> np.ndarray:
		"""
		Computation of Forward Kinematics by classics D-H tables  

		:param th: joints angle
		:returns: 4x4 transformation matrix
		"""
		# D-H parameters of UR3
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

	def run(self):
		t_mat = self.fwd_kinematic(self.th)
		print(t_mat)
		print(np.eye(4))
		posture = self._get_orient(t_mat, np.eye(4))
		print(posture)

if __name__ == "__main__":  
	FK_solver = FK([0, -90, 0, -90, 0, 0])
	FK_solver.run()