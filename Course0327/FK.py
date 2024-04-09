#%% import module
import numpy as np
from numpy import pi, cos, sin, arctan2

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

	def get_posture_diff(self, T_desired: np.ndarray, T_current: np.ndarray) -> np.ndarray:
		"""
		:param T_desired: d-h table of target
		:param T_current: d-h table current state
		:returns: 6x1 array translation + rotation  
		"""
		# m -> mm
		Td = T_desired[:3,3] * 1000 
		Ti = T_current[:3,3] * 1000

		Rd = T_desired[:3,:3]
		Ri = T_current[:3,:3]

		R = self.rot_mat_2_axis_angle(Rd, Ri)
		
		return np.r_[np.array([Td-Ti]).T, R]	

	def run(self):
		t_mat = self.fwd_kinematic(self.th)
		print("Transformation matrix:")
		print(t_mat)
		posture = self.get_posture_diff(t_mat, np.eye(4))
		print("\nTCP posture:")
		print(posture.round(3))

#%% main
np.set_printoptions(suppress=True)
joint_angles = [10, -20, -30, 40, 50, 60]
FK_solver = FK(joint_angles)
FK_solver.run()

# %%
