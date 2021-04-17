from scenario import *
import matplotlib
from matplotlib import animation

pl = plane(np.array([0,0,0,0]), normal=np.array([1,0,0,0]))

rout = routine(f=fun, tInit=0,tFinal=1e4, y0=qp0, ordinaryStepLen=1e-4,method="RK4",timeline=True, savePlaneCuts=True, nopythonExe=True, timelineJumps=300)
rout.planes.append(pl)
y,times,cuts = rout.run()


q1,q2,p1,p2=y.T
q=np.array([q1,q2]).T
p=np.array([p1,p2]).T

import matplotlib.pyplot as plt
plt.plot(q1,q2, linewidth=0.03)
plt.show()

#plt.plot(times, K(p),label="K")
#plt.plot(times, U(q),label="U")
plt.plot(times, H(y),label="H")
plt.legend()
plt.show()

cuts = np.array(cuts[0])
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(1,1)
curve, = ax.plot([],[], "ko", markersize=0.5, label="PoincareCut")
xhat, = ax.plot([],[], "r-", label=r"$q_2$")
yhat, = ax.plot([],[], "g-", label=r"$p_1$")
zhat, = ax.plot([],[], "b-", label=r"$p_2$")
ax.set_xlim([-1.2,1.2])
ax.set_ylim([-1.2,1.2])
ax.set_aspect('equal')
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax.legend(loc="upper right")

@jit(nopython=True)
def project(cutPoint, plposition, plbasis, sourceCrd, dim):
    def normalize(vec): return vec/np.sum(np.sqrt(vec**2))
    plnormal = plbasis[0]
    vector = normalize(cutPoint - sourceCrd)
    if(np.abs(np.sum(plnormal * vector)) < 1e-2):return None, False
    planeDistance = np.sum(plnormal * (plposition-sourceCrd))
    scaling = planeDistance/(np.sum(vector * plnormal))
    #if(scaling < 0): return None, False
    landingPoint = vector * scaling + sourceCrd
    def projectedHyperplaneCoordinates(position):
        relpos = position - plposition
        return np.array([np.dot(plbasis[i + 1], relpos) for i in range(dim - 1)])
    return projectedHyperplaneCoordinates(landingPoint), True

def projectSet(dataSet, plane, source):
    projections = []
    for cutPoint in dataSet:
        var, success = project(cutPoint, plane.position, plane.basis, source, plane.dim)
        if(success) : projections.append(var)
    return np.array(projections)

def init():
    curve.set_data([],[])


frames = 120
r = 1
def expose(i):
    theta = 2*np.pi/frames * i
    normal = np.array([np.cos(theta), np.sin(theta), 0])
    crd = normal*r
    zeta = normal
    eta = np.array([0,0,1])
    xi = cross(np.array([eta, zeta]))
    basis = np.array([zeta,xi,eta])
    pl = plane(-crd, basis=basis)

    xv = projectSet(np.array([[0,0,-0.5],[0.1,0,-0.5]]), pl, crd)
    xhat.set_data(xv.T)
    yv = projectSet(np.array([[0, 0, -0.5], [0, 0.1, -0.5]]), pl, crd)
    yhat.set_data(yv.T)
    zv = projectSet(np.array([[0, 0, -0.5], [0, 0, -0.4]]), pl, crd)
    zhat.set_data(zv.T)

    projectedData = projectSet(cuts, pl, crd)
    curve.set_data(projectedData.T)
    return curve



matplotlib.rcParams['animation.ffmpeg_path'] = r"../../ffmpegLibFiler/bin/ffmpeg.exe";
writer = animation.FFMpegWriter(fps=30);
fa = FuncAnimation(fig, expose, frames=frames,save_count=frames, init_func=init)
#fa.save("fig.gif", fps=30)
fa.save("illustrasjon.mp4",writer=writer,dpi=200)