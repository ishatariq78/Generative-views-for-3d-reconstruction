#!/usr/bin/env python3
"""Is the Sim(3) fit putting the clouds upside down?  shaper env.

The tracker's frame has a known up (camera ring above the object); the model frame is +z up.
For each object: (a) where the unconstrained fit sends tracker-up; (b) the best *upright* fit
(cloud pre-rotated so up=+z, azimuth search + ICP, tilt fixed) with its Chamfer cost and
SLAM-reached, side by side with the unconstrained one.
"""
import sys, json, pickle, zlib, numpy as np, trimesh
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment3/scripts")
import align_and_score as A
from scipy.spatial import cKDTree
EXP3="/home/isha.tariq/viewcrafter_orbits/experiment3/outputs"; SH="/home/isha.tariq/shapeR_testing/data"
OBJ=["ADT1292__chair","BNB1045__sofa","BNB1723__bed","ADT1292__woodentable","BNB2727__bench"]

def rot_to_z(v):
    v=v/np.linalg.norm(v); z=np.array([0,0,1.]); a=np.cross(v,z); s=np.linalg.norm(a); c=v@z
    if s<1e-8: return np.eye(3) if c>0 else np.diag([1,-1,-1.])
    K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])/s; return np.eye(3)+s*K+(1-c)*K@K

out=[]
for obj in OBJ:
    kv=np.load(f"{EXP3}/{obj}__seva360fk5/kv.npz"); cam=kv["traj"][:,:3,3]; pts=kv["object_points"].astype(np.float64)
    ctr=cam.mean(0); n=np.linalg.svd(cam-ctr)[2][-1]; n*=np.sign((ctr-pts.mean(0))@n)
    d=pickle.load(open(f"{SH}/{obj}.pkl","rb")); slam=np.asarray(d["points_model"].numpy(),np.float64)
    mv=np.asarray(d["mesh_vertices"]); diag=float(np.linalg.norm(mv.max(0)-mv.min(0)))
    if len(pts)>120000: pts=pts[np.random.default_rng(zlib.crc32(np.ascontiguousarray(pts))).choice(len(pts),120000,replace=False)]
    reached=lambda al: float((cKDTree(al).query(slam)[0] < 0.02*diag).mean())
    sn,sc,ss=A.normalise(slam)
    # (a) unconstrained, as scored
    M,cost=A.fit(pts,slam); R=M[:3,:3]/np.cbrt(abs(np.linalg.det(M[:3,:3]))); upz=float((R@n)[2])
    al=A.apply(M,A.normalise(pts)[0])*ss+sc; r_free=reached(al)
    # (b) upright only: pre-rotate so tracker-up = +z, then azimuth grid x scale grid + ICP, tilt 0
    pu=(rot_to_z(n)@pts.T).T; src=A.normalise(pu)[0]; tgt=sn; rng=np.random.default_rng(0)
    s_small=src[rng.choice(len(src),min(A.N_COARSE_SRC,len(src)),replace=False)]; tree=cKDTree(tgt)
    best=(np.inf,None)
    for az in np.linspace(0,2*np.pi,24,endpoint=False):
        for s0 in A.SCALE_GRID:
            M0=A.similarity(trimesh.transformations.rotation_matrix(az,[0,0,1]),s0)
            c0=A.sym_chamfer(A.apply(M0,s_small),tgt,tree)
            if c0<best[0]: best=(c0,M0)
    M1,_,_=trimesh.registration.icp(s_small,tgt,initial=best[1],max_iterations=60,scale=True,reflection=False)
    c1=A.sym_chamfer(A.apply(M1,s_small),tgt,tree); R1=M1[:3,:3]/np.cbrt(abs(np.linalg.det(M1[:3,:3])))
    al1=A.apply(M1,src)*ss+sc; r_up=reached(al1); upz1=float((R1@np.array([0,0,1.]))[2])
    print(f"{obj:22s} free fit: cost {cost:.4f} reached {r_free:.3f} up->z {upz:+.2f}   |   upright fit: cost {c1:.4f} reached {r_up:.3f} up->z {upz1:+.2f}")
    out.append(dict(object=obj,free=dict(cost=cost,reached=r_free,up_z=upz),upright=dict(cost=c1,reached=r_up,up_z=upz1)))
json.dump(out,open("outputs/upright_check.json","w"),indent=2)
