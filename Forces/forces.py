#
# Copyright (c) 2023 TITAN Contributors (cf. AUTHORS.md).
#
# This file is part of TITAN 
# (see https://github.com/strath-ace/TITAN).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import numpy as np
from Dynamics.frames import R_B_W
from Dynamics import frames
from scipy.spatial.transform import Rotation as Rot
from Geometry.tetra import vol_tetra

def compute_aerodynamic_forces(titan, options):
    """
    Computes the aerodynamic forces in the wind frame

    Parameters
    ----------
    titan: Assembly_list
        Object of class Assembly_list
    options: Options
        Object of class Options
    """

    if options.vehicle and options.vehicle.Cd:
        for assembly in titan.assembly:
            Aref = options.vehicle.Aref
            Cd = options.vehicle.Cd(assembly.freestream.mach)
            drag = 0.5 *  Aref * Cd * assembly.freestream.density * assembly.freestream.velocity ** 2

            assembly.wind_force.drag = drag

            #The statement below assumes the angle of attack to be always 0
            assembly.body_force.force = np.array([-drag, 0, 0])

    else:
        for assembly in titan.assembly:

            force_facets = -assembly.aerothermo.pressure[:,None]*assembly.mesh.facet_normal+assembly.aerothermo.shear*np.linalg.norm(assembly.mesh.facet_normal, axis=1)[:,None]
            force = np.sum(force_facets, axis = 0)
            assembly.body_force.force = force
            assembly.body_force.force_facets = force_facets
            print('\n\n FORCE:', force)
            q = assembly.quaternion

            R_W_NED = frames.R_W_NED(fpa = assembly.trajectory.gamma, ha = assembly.trajectory.chi)
            R_NED_ECEF = frames.R_NED_ECEF(lat = assembly.trajectory.latitude, lon = assembly.trajectory.longitude)
            R_B_W_quat =  (R_NED_ECEF * R_W_NED).inv()*Rot.from_quat(q)

            #Applies a rotation matrix to change from Body frame to Wind Frame
            aerodynamic_forces = R_B_W_quat.apply(force)*[-1,1,-1]

            assembly.wind_force.drag      = aerodynamic_forces[0]
            assembly.wind_force.crosswind = aerodynamic_forces[1]
            assembly.wind_force.lift      = aerodynamic_forces[2]

def compute_aerodynamic_moments(titan, options):
    """
    Computes the aerodynamic moments in the wind Body frame

    Parameters
    ----------
    titan: Assembly_list
        Object of class Assembly_list
    options: Options
        Object of class Options

    """

    #Computed on the Body Frame

    for assembly in titan.assembly:

        Moment = np.zeros((len(assembly.mesh.facets),3))

        #TODO missing skin friction
        force_facets = -assembly.aerothermo.pressure[:,None]*assembly.mesh.facet_normal+assembly.aerothermo.shear*np.linalg.norm(assembly.mesh.facet_normal, axis=1)[:,None]

        dist = (assembly.mesh.facet_COG[:]-assembly.COG)
        Moment[:,0] = (force_facets[:,2] * dist[:,1] - force_facets[:,1] * dist[:,2]) 
        Moment[:,1] = (force_facets[:,0] * dist[:,2] - force_facets[:,2] * dist[:,0]) 
        Moment[:,2] = (force_facets[:,1] * dist[:,0] - force_facets[:,0] * dist[:,1]) 

        moment = np.sum(Moment, axis = 0)

        assembly.body_force.moment = moment


#Compute inertial forces for FEniCS
def compute_inertial_forces(assembly, options):
    """
    Computes the inertial forces in the Body Frame

    This functions computes the inertial forces that will be used for the Structurla dynamics

    Parameters
    ----------
    assembly: Assembly
        Object of class Assembly
    options: Options
        Object of class Options
    """

    elements = assembly.mesh.vol_elements
    coords = assembly.mesh.vol_coords

    #Computes the voolume and mass of each tetra
    vol = vol_tetra(coords[elements[:, 0]], coords[elements[:, 1]], coords[elements[:, 2]], coords[elements[:, 3]])
    mass = vol * assembly.mesh.vol_density

    # assigning to each vertex 1/4 of the mass of all tetrahedra they belong to
    mass_vertex = np.zeros((coords.shape[0],1))

    for k in range(4):
        np.add.at(mass_vertex, (elements[:,k],0), mass/4)

    angle_vel = np.array([assembly.roll_vel, assembly.pitch_vel, assembly.yaw_vel])
    r_dist = coords - assembly.COG

    #Computes the centrigual acceleration wth respect to the Center of Mass of the assembly
    acc_centrifugal = -np.cross(angle_vel, np.cross(angle_vel, r_dist))

    inertial_loads = mass_vertex * acc_centrifugal

    return inertial_loads


def compute_coefficients(todo):
    pass


def compute_thrust_force(titan, options):
    """
    Computes the thrust force in the body frame
    Parameters
    ----------
    titan: Assembly_list
        Object of class Assembly_list
    options: Options
        Object of class Options
    """


    thrust = np.zeros(3)

    Tmax = 0#40000000
    Tmin = 0#20000000
    t_linear = 5

    if titan.time <= titan.booster_t_trigger:
        T = Tmax
    elif titan.time > titan.booster_t_trigger and titan.time < (titan.booster_t_trigger + t_linear):
        T = Tmax - ((Tmax-Tmin)/t_linear)*(titan.time-titan.booster_t_trigger)
    else:
        T = Tmin

    thrust = [T, 0, 0]

    i = 0
    for assembly in titan.assembly:
        print('Assembly ID =', i, ', Thrust=', assembly.body_force.thrust, ', Moment=', assembly.body_force.thrust_moment)
        i+=1    

    if titan.time == titan.booster_t_trigger:

        print('Booster separation occurring, applying lateral thrust in boosters');

        lateral_boost  = 10000000#4000000
        rotation_boost = 0#10000000
        backward_boost = 0#0.65*lateral_boost

        for assembly in titan.assembly:
            if any(obj.name == 'Tests/Mesh/Ariane/cubes/UpperSection_M.stl' for obj in assembly.objects):
            #if any(obj.name == 'Tests/Mesh/Ariane/UpperSection_M.stl' for obj in assembly.objects):
                assembly.body_force.thrust = thrust
            elif any(obj.name == 'Tests/Mesh/Ariane/cubes/RadialA_M.stl' for obj in assembly.objects):
            #elif any(obj.name == 'Tests/Mesh/Ariane/RadialA_M.stl' for obj in assembly.objects):
                assembly.body_force.thrust = [-backward_boost,lateral_boost,0]
                assembly.body_force.thrust_moment = [0,0,-rotation_boost]        
            elif any(obj.name == 'Tests/Mesh/Ariane/cubes/RadialB_M.stl' for obj in assembly.objects):
            #elif any(obj.name == 'Tests/Mesh/Ariane/RadialB_M.stl' for obj in assembly.objects):
                assembly.body_force.thrust = [-backward_boost,-lateral_boost,0]
                assembly.body_force.thrust_moment = [0,0,rotation_boost+0.75*rotation_boost]
    else:

        for assembly in titan.assembly:
            if any(obj.name == 'Tests/Mesh/Ariane/ArianeBody_Euler.stl' for obj in assembly.objects):
                assembly.body_force.thrust = thrust
            else:
                assembly.body_force.thrust        = np.zeros(3)
                assembly.body_force.thrust_moment = np.zeros(3)
    i = 0
    for assembly in titan.assembly:
        print('Assembly ID =', i, ', Thrust=', assembly.body_force.thrust, ', Moment=', assembly.body_force.thrust_moment)
        i+=1
