# Derivation
> This derivation is used in V2

## Maxwell Equation
```math
\nabla\times E=-\mu{\partial H \over \partial t} \\
\nabla\times H=\epsilon {\partial E \over \partial t}
```

## Normalized
```math
\nabla\times E = \mu_r H \\
\nabla\times H = \epsilon_r E
```

### Normalization (old -> new)
```math
i\sqrt{\mu_0\over\epsilon_0} H \rightarrow H \\
k_0 (x, y, z) \rightarrow (x, y, z)
```

## Matrix Form
```math
\begin{bmatrix}
0 & -\partial_z & \partial_y \\
\partial_z & 0 & -\partial_x \\
-\partial_y & \partial_x & 0
\end{bmatrix}
\begin{bmatrix} E_x \\ E_y \\ E_z \end{bmatrix}
= \mu_r \begin{bmatrix} H_x \\ H_y \\ H_z \end{bmatrix}
```

```math
\begin{bmatrix}
0 & -\partial_z & \partial_y \\
\partial_z & 0 & -\partial_x \\
-\partial_y & \partial_x & 0
\end{bmatrix}
\begin{bmatrix} H_x \\ H_y \\ H_z \end{bmatrix}
= \epsilon_r \begin{bmatrix} E_x \\ E_y \\ E_z \end{bmatrix}
```

### Assumption
```math
U = \sum_m U_m(z) \exp(i\vec k_m \cdot \vec r) \\
k_m = \vec k_{in} + \vec G_m
```
where $U$ can be $E_x, E_y, E_z, H_x, H_y, H_z$.

### Expand
The $E(x, y, z)$ becomes $E(z)$, and we calculate the partial derivative using the assumption above.

Also, we consider the full matrix including all tensor terms.

```math
\begin{bmatrix}
0 & -({d\over dz}+ikz) & iky \\
({d\over dz}+ikz) & 0 & -ikx \\
-iky & ikx & 0
\end{bmatrix}
\begin{bmatrix} E_x \\ E_y \\ E_z \end{bmatrix}
= 
\begin{bmatrix}
\mu_{xx} & \mu_{xy} & \mu_{xz} \\
\mu_{yx} & \mu_{yy} & \mu_{yz} \\
\mu_{zx} & \mu_{zy} & \mu_{zz}
\end{bmatrix}
\begin{bmatrix} H_x \\ H_y \\ H_z \end{bmatrix}
```

### XY Part (First Two Rows)
```math
({d\over dz}+ik_z)
\begin{bmatrix}
0 & -1 \\
1 & 0 
\end{bmatrix}
\begin{bmatrix} E_x \\ E_y \end{bmatrix}
+
\begin{bmatrix} ik_y \\ -ik_x \end{bmatrix} 
E_z 
= 
\begin{bmatrix}
\mu_{xx} & \mu_{xy} \\ 
\mu_{yx} & \mu_{yy}
\end{bmatrix}
\begin{bmatrix} H_x \\ H_y \end{bmatrix}
+
\begin{bmatrix} \mu_{xz} \\ \mu_{yz} \end{bmatrix}
H_z 
```

#### Solve
```math
({d\over dz}+ik_z)
\begin{bmatrix}
0 & -1 \\
1 & 0 
\end{bmatrix}
\begin{bmatrix} E_x \\ E_y \end{bmatrix}
= 
-
\begin{bmatrix} ik_y \\ -ik_x \end{bmatrix} 
E_z 
+
\begin{bmatrix}
\mu_{xx} & \mu_{xy} \\ 
\mu_{yx} & \mu_{yy} 
\end{bmatrix}
\begin{bmatrix} H_x \\ H_y \end{bmatrix}
+
\begin{bmatrix} \mu_{xz} \\ \mu_{yz} \end{bmatrix}
H_z 
```

Organize into:

> #### $E_x, E_y$ Part
> ```math
> {d\over dz}
> \begin{bmatrix} E_x \\ E_y \end{bmatrix}
> = 
> -ik_z
> \begin{bmatrix} E_x \\ E_y \end{bmatrix}
> +
> \begin{bmatrix} ik_x \\ ik_y \end{bmatrix} 
> E_z 
> +
> \begin{bmatrix}
> \mu_{yx} & \mu_{yy} \\
> -\mu_{xx} & -\mu_{xy}  
> \end{bmatrix}
> \begin{bmatrix} H_x \\ H_y \end{bmatrix}
> +
> \begin{bmatrix} \mu_{yz} \\ -\mu_{xz} \end{bmatrix}
> H_z 
> ```

> #### $H_x, H_y$ Part
> ```math
> {d\over dz}
> \begin{bmatrix} H_x \\ H_y \end{bmatrix}
> = 
> -ik_z
> \begin{bmatrix} H_x \\ H_y \end{bmatrix}
> +
> \begin{bmatrix} ik_x \\ ik_y \end{bmatrix} 
> H_z 
> +
> \begin{bmatrix}
> \epsilon_{yx} & \epsilon_{yy} \\
> -\epsilon_{xx} & -\epsilon_{xy}  
> \end{bmatrix}
> \begin{bmatrix} E_x \\ E_y \end{bmatrix}
> +
> \begin{bmatrix} \epsilon_{yz} \\ -\epsilon_{xz} \end{bmatrix}
> E_z 
> ```

### Z Part (Last Row)
```math
\begin{bmatrix}
-ik_y & ik_x
\end{bmatrix}
\begin{bmatrix} E_x \\ E_y \end{bmatrix}
= 
\begin{bmatrix}
\mu_{zx} & \mu_{zy} 
\end{bmatrix}
\begin{bmatrix} H_x \\ H_y \end{bmatrix}
+
\mu_{zz}
H_z
```

Solve for $z$, organize into:

> #### $H_z$ Part
> ```math
> H_z = 
> \mu_{zz}^{-1}
> \begin{bmatrix}
> -ik_y & ik_x
> \end{bmatrix}
> \begin{bmatrix} E_x \\ E_y \end{bmatrix}
> -
> \mu_{zz}^{-1}
> \begin{bmatrix}
> \mu_{zx} & \mu_{zy} 
> \end{bmatrix}
> \begin{bmatrix} H_x \\ H_y \end{bmatrix}
> ```

> #### $E_z$ Part
> ```math
> E_z = 
> \epsilon_{zz}^{-1}
> \begin{bmatrix}
> -ik_y & ik_x
> \end{bmatrix}
> \begin{bmatrix} H_x \\ H_y \end{bmatrix}
> -
> \epsilon_{zz}^{-1}
> \begin{bmatrix}
> \epsilon_{zx} & \epsilon_{zy} 
> \end{bmatrix}
> \begin{bmatrix} E_x \\ E_y \end{bmatrix}
> ```

## Final Form

Put $E_z, H_z$ into $E_x, E_y, H_x, H_y$ equations.

#### $E_x, E_y$ Part
```math
{d\over dz}
\begin{bmatrix} E_x \\ E_y \end{bmatrix}
=\\
\left(
    -ik_z
    -
    \begin{bmatrix} ik_x \\ ik_y \end{bmatrix} 
    \epsilon_{zz}^{-1}
    \begin{bmatrix}
    \epsilon_{zx} & \epsilon_{zy} 
    \end{bmatrix}
    +
    \begin{bmatrix} \mu_{yz} \\ -\mu_{xz} \end{bmatrix}
    \mu_{zz}^{-1}
    \begin{bmatrix}
    -ik_y & ik_x
    \end{bmatrix}
\right)
\begin{bmatrix} E_x \\ E_y \end{bmatrix}
+\\
\left(
    \begin{bmatrix} ik_x \\ ik_y \end{bmatrix} 
    \epsilon_{zz}^{-1}
    \begin{bmatrix}
    -ik_y & ik_x
    \end{bmatrix}
    +
    \begin{bmatrix}
    \mu_{yx} & \mu_{yy} \\
    -\mu_{xx} & -\mu_{xy}  
    \end{bmatrix}
    -
    \begin{bmatrix} \mu_{yz} \\ -\mu_{xz} \end{bmatrix}
    \mu_{zz}^{-1}
    \begin{bmatrix}
    \mu_{zx} & \mu_{zy} 
    \end{bmatrix}
\right)
\begin{bmatrix} H_x \\ H_y \end{bmatrix} 
```

#### $H_x, H_y$ Part
```math
{d\over dz}
\begin{bmatrix} H_x \\ H_y \end{bmatrix}
=\\
\left(
    -ik_z
    -
    \begin{bmatrix} ik_x \\ ik_y \end{bmatrix} 
    \mu_{zz}^{-1}
    \begin{bmatrix}
    \mu_{zx} & \mu_{zy} 
    \end{bmatrix}
    +
    \begin{bmatrix} \epsilon_{yz} \\ -\epsilon_{xz} \end{bmatrix}
    \epsilon_{zz}^{-1}
    \begin{bmatrix}
    -ik_y & ik_x
    \end{bmatrix}
\right)
\begin{bmatrix} H_x \\ H_y \end{bmatrix}
+\\
\left(
    \begin{bmatrix} ik_x \\ ik_y \end{bmatrix} 
    \mu_{zz}^{-1}
    \begin{bmatrix}
    -ik_y & ik_x
    \end{bmatrix}
    +
    \begin{bmatrix}
    \epsilon_{yx} & \epsilon_{yy} \\
    -\epsilon_{xx} & -\epsilon_{xy}  
    \end{bmatrix}
    -
    \begin{bmatrix} \epsilon_{yz} \\ -\epsilon_{xz} \end{bmatrix}
    \epsilon_{zz}^{-1}
    \begin{bmatrix}
    \epsilon_{zx} & \epsilon_{zy} 
    \end{bmatrix}
\right)
\begin{bmatrix} E_x \\ E_y \end{bmatrix} 
```
