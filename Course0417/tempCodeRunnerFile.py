	if(np.any(q[-1,:] > pi) or np.any(q[iter+1,:] < -pi)): 
				l = np.argwhere(q[-1,:] > pi)
				k = np.argwhere(q[-1,:] < -pi)
				q[-1,l] = q[iter,l]
				q[-1,k] = q[iter,k]