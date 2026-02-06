# ArrayGen
###### Version 1.3.3
____
~ Aarin J
____

ArrayGen is an array generation library which is intended for students.
In its bare bones, it creates a domain-range pair.
____
> Currently requires numpy and matplotlib

###### ! Do note that all functions are not documented. They will in due time.

## Modules:
Firstly, let's have a look at all the modules:

```commandline
.Physics
.Maths
.Graphing
.utils
.inb_
```
###### These names are very self-explanatory, but nevertheless, let's dive deeper.
____
## `.Physics`
As the name says, this module relates to physics.
It includes:
```commandline
projectile_motion()
attraction()
dot()
cross()
```
____
`projectile_motion()`

This function uses:
- u: float, 

This is the initial velocity.
- theta_degrees: float,

Angle
- range_: Optional[float] = None,

Optional range (if provided, can find height)
- height_: Optional[float] = None,

Optional height (if provided, can find range)
- f_range: bool = False,

Finds range
- f_height: bool = False,

Finds height
- gravity: float = 9.8,

Gravity constant (can be changed)
- graph: bool = True,

Whether to plot a graph of the trajectory
- grid: bool = True,

Whether to add a grid to that graph
- graph_range_x: float = 50,

Sets the x-axis limit / range
- graph_range_y: float = 50,

Sets the y-axis limit / range

>Example:
```python
import ArrayGen as q

u = 50 # Initial velocity = 50 m/s
theta = 30 # Angle = 30 degrees

# With the graph
q.Physics.projectile_motion(u, theta)
# Graph is True by default. You can optionally add ...(..., grid=False)

# Without graph
x, y = q.Physics.projectile_motion(u, theta, graph=False)
print(x, y)
```

The function uses classical equations of projectile motion, **assuming no air resistance**:

Formulae: 
$$
y = x \\tan(\theta) - \frac{g x^2}{2 u^2 \cos^2(\theta)}
$$
This is for the projectile trajectory.

$$
R = \frac{u^2 \sin(2\theta)}{g}
$$

$$
H = \frac{u^2 \sin^2(\theta)}{2g}
$$

And these are for the range and height respectively.
____
`attraction()`

This is a function which creates an interactive vector field. Wherein the user's mouse (the magnet) is attracting the vectors.
This, however doesn't work in .ipynb files.

____
`dot_product()`

This function uses:

- expr1: str

The first vector.
- expr2: str

The second vector.

These vectors are as strings as the `get_num()` function is used to extract numerical values in string datatypes. This will be seen more.

> Example:
```python
import ArrayGen as q

expr1 = '3i+2j-4k'
expr2 = '4i+2j+1k'

ans = q.Physics.dot_product(expr1, expr2)
print(ans)
```

Formulae:

Say we have 2 vectors:

$$A = ai + bj + ck$$

$$B = xi + yj + zk$$

Then,

$$
A . B = ax + by + zc
$$

$$
A . B = |A| |B| cos\theta
$$
But here, this function is only using the first formula in later versions.*

> ###### *Will implement the second formula.
____

## `.Maths`
Contents in `.Maths`:

> hypo() \
> pascal_triangle() \
> mandelbrot_set()

____

> Complex
>> z_algebra() \
>> z_forms()

____

> Progressions
>> term() \
>> get_progression()


____

`hypo()`

As it says, this function is used to find the hypotenuse of any triangle.

The function uses:

- a: float

The magnitude of side 1.
- b: float

The magnitude of side 2.
- angle_A: float

The angle of the triangle.