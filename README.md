I have constructed a Python code to design and synthesize an arbitrary digital
optical moving average(MA) bandpass filter using the lattice architecture. Using this
architecture, the filter can be constructed by cascading Mach-Zehnder interferometers(
MZI’s). The code is capable of designing filters with multiple pass or stop
bands. This type of filter is also called a finite impulse response(FIR) filter. The
program makes extensive use of NumPy’s C-implemented array structures to greatly
speed up computations.
The program uses Parks-McClellan algorithm methods to determine the Nth- order
FIR transfer function AN(z).

The Parks-McClellan algorithm is an optimization algorithm that seeks to minimize
an objective function, in this case the frequency response of the filter, subject to
the constraint of the maximum deviation from the “ideal” frequency response in the
pass and stop bands. Therefore, such a filter is called a equiripple filter. Finding
the filter coefficients ultimately comes down to solving the system of equations. The
alternation points and the corresponding solution to the system of equations are
found using an iterative procedure called the Remez exchange algorithm.

It is well known that in order to design a half band filter, we need to have a odd
number of filters. Although in optical half band filter designed by MZI the number
of stages should be even. So I used a trick to the design a zero-phase FIR equiripple half-band filter.
