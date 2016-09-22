import numpy as np
from numpy import *

def find_error_a(Nx,tol,s):

	""" Introducing coefficients: """

	w = 1.0					# Frequency, set to 1.0 for simplicity
	L = 1.0					# Length, ----- " ------
	c = lambda x: sqrt(1 + (x - L/2.0)**4)	# c value equals to sqrt(q)
	C = 1.0					# Relationship between dt & dx				# Mesh points for x-array
	c_max = c(L)				# Since we now use c as a function, we get the maximum at one of the endpoints
	dt = C*((L/2)/Nx)/c_max		# dt is set by this formula
	T = 5					# how far we should go in time

	""" Introducing the functions: """

	u_exact = lambda x,t: cos(  pi * x / float(L)  ) * cos(  w * t  )
	I = lambda x: u_exact(x,0)
	V = lambda x: 0
	f = lambda x,t: ( -w**2 + (pi/L)**2 * (1 + (x-L/2.0)**4) ) * u_exact(x,t) + pi/L * (4*(x - L/2)**3) * sin(pi*x/float(L))*cos(w*t)

	U_0 = None				# Setting U_0 to None implies neumann conditions
	U_L = None				# Setting U_L to None implies neumann conditions

	def assert_no_error(u,x,t,n):
		u_e = u_exact(x, t[n])
		diff = np.abs(u - u_e).max()
		assert diff < tol
	s(I, V, f, c, U_0, U_L, L, dt, C, T,user_action=assert_no_error, version='vectorized', stability_safety_factor=1)
	print " test succeeded! "


def find_error_b(Nx,tol,s):

	""" Introducing coefficients: """

	w = 1.0					# Frequency, set to 1.0 for simplicity
	L = 1.0					# Length, ----- " ------
	c = lambda x: sqrt( 1 + cos(pi * x/L))	# c value equals to sqrt(q)
	C = 1.0					# Relationship between dt & dx
	c_max = c(0)				# Since we now use c as a function, we get the maximum at x = 0 (sqrt(2))
	dt = C*((L/2)/Nx)/c_max			# dt is set by this formula
	T = 5					# how far we should go in time
				# how far we should go in time

	""" Introducing the functions: """

	u_exact = lambda x,t: cos(  pi * x / float(L)  ) * cos(  w * t  )	# Still keeping this u_exact
	I = lambda x: u_exact(x,0)						# The same
	V = lambda x: 0								# The same
	f = lambda x,t: -w**2*cos(t*w)*cos(pi*x/L) + pi**2*(cos(pi*x/L) + 1)*cos(t*w)*cos(pi*x/L)/L**2 - pi**2*sin(pi*x/L)**2*cos(t*w)/L**2
			#((pi/L)**2 - w**2 )*u_exact(x,t) + 2*(pi/L)**2 * cos(w*t)*cos(2*pi*x/L)		# Changed with q-value
	U_0 = None				# Setting U_0 to None implies neumann conditions
	U_L = None				# Setting U_L to None implies neumann conditions

	def assert_no_error(u,x,t,n):
		u_e = u_exact(x, t[n])
		diff = np.abs(u - u_e).max()
		assert diff < tol
	s(I, V, f, c, U_0, U_L, L, dt, C, T,user_action=assert_no_error, version='vectorized', stability_safety_factor=1)
	print " test succeeded! "

def find_conv_rate(Nx1,E1,Nx2,E2):
		return log(E2/float(E1)) / log(Nx1/float(Nx2))

print''
print'Exercise 13:'
print''
print'a)'
print'testing the lowest error i may get with 3 significant digits..'


###
### Exercise a)
###
Nx1 = 100; tol1 = .272*1e-4
Nx2 = 150; tol2 = .121*1e-4

from Solver import solver, solver_a

find_error_a(Nx1,tol1,solver_a)		# Not a good way, but I just tried as low i could go for 3 significant digits, manually
find_error_a(Nx2,tol2,solver_a)		# --- " ---
print'Nx1 = %s -> error1 < %s' %(Nx1,tol1)
print'Nx2 = %s -> error2 < %s' %(Nx2,tol2)
print''
print "The convergence rate for problem a) is: r =  %s"  %find_conv_rate(Nx1,tol1,Nx2,tol2)
print''
print'b)'

###
### Exercise b)
###

from Solver import solver_b

Nx1 = 100; tol1 = .227*1e-2		# Not a good way, but I just tried as low i could go for 3 significant digits, manually
Nx2 = 150; tol2 = .119*1e-2		# ---- " ----

find_error_b(Nx1,tol1,solver_b)
find_error_b(Nx2,tol2,solver_b)


print'Nx1 = %s -> error1 < %s' %(Nx1,tol1)
print'Nx2 = %s -> error2 < %s' %(Nx2,tol2)
print''
print "The convergence rate for problem b) is: r =  %s" %find_conv_rate(Nx1,tol1,Nx2,tol2)
print''
print'c)'

###
### Exercise c)
###

from Solver import solver_c

Nx1 = 100; tol1 = .235*1e-1		# Not a good way, but I just tried as low i could go for 3 significant digits, manually
Nx2 = 150; tol2 = .157*1e-1		# ---- " ----

find_error_a(Nx1,tol1,solver_c)
find_error_a(Nx2,tol2,solver_c)

print'Nx1 = %s -> error1 < %s' %(Nx1,tol1)
print'Nx2 = %s -> error2 < %s' %(Nx2,tol2)
print''
print "The convergence rate for problem c) is: r = %s" %find_conv_rate(Nx1,tol1,Nx2,tol2)
print ''
print ''
print 'Conclusion:'
print 'We expect the rate to be around 2, especially in a)'
print 'After some modification in b), the convergence rate decreases'
print 'In c) we use a even worse approximation and expect the convergence rate to be low'
print ''





