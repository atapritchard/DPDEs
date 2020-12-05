# Credit to G. Nervadof
# Source: https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
# Taken on December 3rd, 2020
# Solver for 2-D Heat Equation with GIF outputZ
# Article text
'''
When I was in college studying physics a few years ago, I remember there was a task to solve heat equation analytically for some simple problems. In the next semester we learned about numerical methods to solve some partial differential equations (PDEs) in general. It’s really interesting to see how we could solve them numerically and visualize the solutions as a heat map, and it’s really cool (pun intended). I also remember, in the previous semester we learned C programming language, so it was natural for us to solve PDEs numerically using C although some students were struggling with C and not with solving the PDE itself. If I had known how to code in Python back then, I would’ve used it instead of C (I am not saying C is bad though). Here, I am going to show how we can solve 2D heat equation numerically and see how easy it is to “translate” the equations into Python code.

Before we do the Python code, let’s talk about the heat equation and finite-difference method. Heat equation is basically a partial differential equation, it is
Image for post
Image for post

If we want to solve it in 2D (Cartesian), we can write the heat equation above like this
Image for post
Image for post

where u is the quantity that we want to know, t is for temporal variable, x and y are for spatial variables, and α is diffusivity constant. So basically we want to find the solution u everywhere in x and y, and over time t.

Now let’s see the finite-difference method (FDM) in a nutshell. Finite-difference method is a numerical method for solving differential equations by approximating derivative with finite differences. Remember that the definition of derivative is
Image for post
Image for post

In finite-difference method, we approximate it and remove the limit. So, instead of using differential and limit symbol, we use delta symbol which is the finite difference. Note that this is oversimplified, because we have to use Taylor series expansion and derive it from there by assuming some terms to be sufficiently small, but we get the rough idea behind this method.
Image for post
Image for post

In finite-difference method, we are going to “discretize” the spatial domain and the time interval x, y, and t. We can write it like this
Image for post
Image for post
Cartesian coordinate, where x and y axis are for spatial variables, and t for temporal variable (coordinate axes from GeoGebra, edited by author)
Image for post
Image for post
Image for post
Image for post
Image for post
Image for post

As we can see, i, j, and k are the steps for each difference for x, y, and t respectively. What we want is the solution u, which is
Image for post
Image for post

Note that k is superscript to denote time step for u. We can write the heat equation above using finite-difference method like this
Image for post
Image for post

If we arrange the equation above by taking Δx = Δy, we get this final equation
Image for post
Image for post

where
Image for post
Image for post

We can use this stencil to remember the equation above (look at subscripts i, j for spatial steps and superscript k for the time step)
Image for post
Image for post
Explicit method stencil (image by author)

We use explicit method to get the solution for the heat equation, so it will be numerically stable whenever
Image for post
Image for post

Everything is ready. Now we can solve the original heat equation approximated by algebraic equation above, which is computer-friendly. For an exercise problem, let’s suppose a thin square plate with the side of 50 unit length. The temperature everywhere inside the plate is originally 0 degree (at t = 0), let’s see the diagram below (this is not realistic, but it’s good for exercise)
Image for post
Image for post
Boundary and initial conditions for our exercise (image by author)

For our model, let’s take Δx = 1 and α = 2.0. Now we can use Python code to solve this problem numerically to see the temperature everywhere (denoted by i and j) and over time (denoted by k). Let’s first import all of the necessary libraries, and then set up the boundary and initial conditions.

We’ve set up the initial and boundary conditions, let’s write the calculation function based on finite-difference method that we’ve derived above.

Let’s prepare the plot function so we can visualize the solution (for each k) as a heat map. We use Matplotlib library, it’s easy to use.

One more thing that we need is to animate the result because we want to see the temperature points inside the plate change over time. So let’s create the function to animate the solution.

Now, we’re done! Let’s see the complete code below and run it.

That’s it! And here’s the result
Image for post
Image for post
The numeric solution of our simple heat equation exercise

Cool isn’t it? By the way, you can try the code above using this Python Online Compiler https://repl.it/languages/python3, make sure you change the max_iter_time to 50 before you run the code to make the iteration result faster.

Okay, we have the code now, let’s play with something more interesting, let’s set all of the boundary conditions to 0, and then randomize the initial condition for the interior grid.

And here’s the result
Image for post
Image for post
The numeric solution where all boundary conditions are 0 with randomized initial condition inside the grid

Python is relatively easy to learn for beginners compared to other programming languages. I would recommend to use Python for solving computational problems like we’ve done here, at least for prototyping because it has really powerful numerical and scientific libraries. The community is also growing bigger and bigger, and that can make things easier to Google when we’re stuck with it. Python may not be as fast as C or C++, but using Python we can focus more on the problem solving itself rather than the language, of course we still need to know the Python syntax and its simple array manipulation to some degree, but once we get it, it will be really powerful.

'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

from pdb import set_trace as debug

print("2D heat equation solver")

plate_length = 100
max_iter_time = 400

alpha = 0.1
delta_x = 1

delta_t = (delta_x ** 2)/(4 * alpha)
gamma = (alpha * delta_t) / (delta_x ** 2)

# Initialize solution: the grid of u(k, i, j)
u = np.empty((max_iter_time, plate_length, plate_length))

# Change boundary conditions
u_top = 0.0
u_left = 0.0
u_bottom = 0.0
u_right = 0.0

# Change u_initial (random temperature between 28.5 and 55.5 degree)
# u_initial = 0
# u_initial = np.random.normal(0, 1, size=(plate_length,plate_length))
u_initial = np.zeros((plate_length, plate_length))
gaussian_pdf = lambda x, y: np.exp(-(x**2+y**2) / 2*0.5)  # |covariance matrix| = 0.5
mid = plate_length // 2
for i in range(plate_length):
    for j in range(plate_length):
        u_initial[i, j] = 200*gaussian_pdf((i - mid) / (mid / 4), (j - mid) / (mid / 4))


# Change initial conditions
# u.fill(u_initial)
u[0, :, :] = u_initial


# Set the boundary conditions
u[:, (plate_length-1):, :] = u_top
u[:, :, :1] = u_left
u[:, :1, 1:] = u_bottom
u[:, :, (plate_length-1):] = u_right


def calculate(u):
    for k in tqdm(range(0, max_iter_time-1, 1), desc='progress', ncols=80):
        for i in range(1, plate_length-1, delta_x):
            for j in range(1, plate_length-1, delta_x):
                u[k + 1, i, j] = gamma * (u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1] - 4*u[k][i][j]) + u[k][i][j]

                # Rework to use scipy solver

    return u

def plotheatmap(u_k, k):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Temperature at t = {k*delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()

    return plt

# Do the calculation here
u = calculate(u)
temp = u

def animate(k):
    plotheatmap(u[k], k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=max_iter_time, repeat=False)
anim.save("heat_equation_solution.gif")

print("Done!")
