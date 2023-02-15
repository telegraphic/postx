# Post-correlation beamforming and direct imaging

This repository contains code to do post-correlation beamforming and imaging. It is designed for compact low-frequency arrays like the EDA2.

<p align="center">
<img src="https://raw.githubusercontent.com/telegraphic/eda_post_x/main/docs/example-allsky.png?token=GHSAT0AAAAAABM2MM5N7EEZFTAQEZIBDLHGY7MUUUA" width="400"/>
</p>

## Introduction

A beamformed *voltage beam* is given by the sum of weights $w=Ae^{i\theta}$
with a voltage stream $v(t)$:

$$
b(t)=\sum_{p=1}^{P}w_{p}v_{p}(t)
$$ 

The *power beam* is the voltage beam after squaring and averaging in time. when averaged with time (denote with $\langle \rangle$ brackets) is:

$$
B  = \langle b(t)b^{H}(t) \rangle \ =\  \boldsymbol{w}_p \boldsymbol{V}_{pq} \boldsymbol{w}_q^H \ \equiv\  \boldsymbol{W}_{pq} \boldsymbol{V}_{pq}^H
$$

Where $\boldsymbol{V}$ is the $(P\times P)$ visibility matrix, $\boldsymbol{w}$ is a $(1\times P)$ weights vector. Equivalently, $\boldsymbol{W}$ is a  $(P\times P)$ weights matrix. The subscripts $p$ and $q$ are row/column indices representing antenna pairs in the matrix. 

In terms of computations, the most efficient way to form a power beam, with order $O(P)$, is first form a voltage beam, square the output, then average. Forming the visibility matrix is an $O(P^2)$ operation, so post-correlation beamforming is much more computationally expensive. The weight matrix, 

$$
\boldsymbol{W}_{pq} = \boldsymbol{w}_{p} \boldsymbol{w}_{q}^H
$$

is conceptually useful (and useful for visualization), but comes at even more computational expense and memory requirements. 

However, post-correlation beamforming is incredibly flexible: a user can form as many beams as they feel like, and can re-point the beam in any desired direction. Also, as interferometers natively output visbility matrices, post-correlation beamforming can be a useful approach if a real-time voltage beamformer does not exist.


## Summation notation and beam grids

The post-correlation approach allows a compact summation notation to forming a grid of beams on the sky (coordinate subscripts $i$ and $j$), across frequency channels (subscript $\nu$):

$$
B_{i j \nu} = W^{p q}_{i j \nu} \, V_{p q \nu}
$$

Summation here is implied over all indices that appear in an upper and lower index. So, we sum across indexes $p$ and $q$ (summation indices), and output an N-dimensional matrix with indices $(i, j, \nu)$. In slightly-less compact, but more efficient form:

$$
B_{i j \nu} = w_{i j p \nu} \, V^{p q}_{\nu} (w^H)_{i j q \nu} (1)
$$

The visibility matrix can itself be written in summation notation as

$$
V_{p q} = v^{t}_{p} (v^H)_{q t} 
$$

where $t$ is summation over time step (instead of using $\langle \rangle$ brackets).

#### Numpy's `einsum`
In Python (Numpy and Cupy), there is a `einsum` command, which has slightly different syntax. The visibility matrix is compute via:

```python
# v: voltage array with shape (N_ant, N_timesteps), complex dtype
# V: visibility matrix with shape (N_ant, N_ant), complex dtype
V = np.einsum('pt,qt->pq', v, v.conj(), optimize=True)
```

For a given frequency $\nu$, a beam matrix is formed via:

```python
# weights: weight array with shape (N_pix, N_pix, N_ant), complex dtype
# V: visibility matrix with shape (N_ant, N_ant), complex dtype
B = np.einsum('ijp,pq,ijq->ij', weights, V, weights.conj(), optimize=True)
```

Note that the `optimize=True` argument is key to good performance, by optimizing the contraction order can dispatch to BLAS packages ([see opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/)).

## Calculating array weights

To phase up an array, we need to generate a set of geometric time delays, $\tau$. Each time delay has a corresponding phase delay:

$$
\phi_p = 2\pi \nu \tau_p
$$

The geometric delay is measured from a reference antenna, which we will set to be $p=0$. To find the geometric delays for a desired pointing direction (target), we first find the baseline lengths between the reference and all other antennas:

$$
(\Delta X_p, \Delta Y_p, \Delta Z_p) = (X_0, Y_0, Z_0) - (X_p, Y_p, Z_p)
$$

1. Use $(l, m)$ direction cosines of the target from the phase center. Useful for making 2D images around the phase center. Remember $(l, m)$ are cartesian coordinates, not spherical.

2. Use the hourangle and declination $(H, d)$ of the target from the phase center. Useful for pointing at a Celestial source with known RA/DEC, or za/az. 

In both instances, the geometric delay is found from the W-component:

$$
w_p = (l, m, \sqrt{1 - l^2 -w^2}) . (\Delta X_p, \Delta Y_p, \Delta Z_p)_{local}
$$

$$
w_p = (cos(d)cos(H), -cos(d)sin(H), sin(d)) . (\Delta X_p, \Delta Y_p, \Delta Z_p)_{celestial}
$$

## A1. Coordinate system recap

### Antenna coordinates (X, Y, Z):

* Local XYZ: Relative to the surface of the Earth, at the observatory site. Normally X=East, Y=North, Z=Up. That is: The Z-direction points toward zenith.

* ECEF XYZ: Abstract coordinates relative to the center of the Earth. The Z direction points toward the the North celestial pole**. The X-axis is in the plane of the Equator and runs through $0^\circ$ and $180^\circ$ degrees. The Y-axis runs through $90^\circ$ W to $90^\circ$ E.

** It's actually the 'international reference pole (IRP)', which I'm not 100% sure is the same as the NCP.

### Spherical coordinates (r, $\theta$, $\phi$):

This is where you are pointing.

* Local za/az: zenith angle (0 to $90^\circ$ ) and azimuthal angle (full 360 degrees). Also called horizontal coordinates.

* Celestial RA/DEC: Right ascension and declination, as measured from the North Celestial pole.

### Interferometer coordinates (UVW):

When making an image with an interferometer, we have to project the 3D celestial sphere into points on a 2D plane.  To do so, we choose a phase center pointing $(\theta_0, \phi_0)$, which is defined as the centre of the 'UVW' coordinate system. 

In UVW coordinates, The W-axis is aligned along the pointing direction. It is **extremely** important to realize that later on the W-component of a baseline *is* equivalent to the geometric delay due to light travel time. The U and V axes are then in a plane.

* Local UVW: The W-direction is chosen to be zenith, and then U and V axes can be lined up with local XY coordinates.

* Celestial UVW: The W-direction is still chosen to be zenith, but we align the XY-plane with the North celestial pole. 

### Direction cosines $(l, m)$ 

When projecting from 3D spherical to 2D, the two plane axes are called $l$ and $m$ are they are direction cosines:
* $l = cos(X)$ 
* $m = cos(Y)$

Which both range from (-1, 1). Note these are cartesian, not spherical coordinates! The phase center is at the origin (0, 0).  The pointing vector that corresponds to this is:

$$
\boldsymbol{s} = (l, m, \sqrt{1 - l^2 - m^2})
$$




