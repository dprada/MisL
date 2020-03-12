import molmodmt as m3t
import simtk.openmm as mm
import simtk.unit as unit
import simtk.openmm.app as app
import numpy as np
from sys import stdout
from sys import exit

from molmodmt.utils import openmm as m3t_openmm

#system_modeller_original = m3t.convert('MisL_Phyre_minimized.pdb','openmm.Modeller')
system_modeller_original = m3t.fix_pdb_structure('MisL_Phyre.pdb',output_form='openmm.Modeller')
minimized_positions = m3t.get(system_modeller_original, coordinates=True)

system_modeller = m3t.convert('fin.pdb','openmm.Modeller')
initial_positions = m3t.get(system_modeller, coordinates=True)
topology = m3t.convert(system_modeller,'openmm.Topology')
forcefield = app.ForceField('amber99sbildn.xml')
system = forcefield.createSystem(topology, constraints=app.HBonds)

## Harmonic restraint absolute positions

atoms_restrained = m3t.select(topology, 'resid 183 to 504')
harmonic_constant = 20.0*unit.kilocalories_per_mole/unit.angstrom**2
pinned_positions = [initial_positions[ii]._value for ii in atoms_restrained]*initial_positions.unit
m3t_openmm.add_harmonic_restraint_in_absolute_positions(system,
                                                        atoms_restrained,
                                                        harmonic_constant,
                                                        pinned_positions)

## Harmonic restraint distances

atoms_sel = m3t.select(topology, 'resid 8 to 169 and name CA')
atoms_pairs = [ [atoms_sel[0],ii] for ii in atoms_sel[1:] ]
atoms_pairs.extend([ [atoms_sel[-1],ii] for ii in atoms_sel[1:-1] ])
harmonic_constant = 1.0 * unit.kilocalories_per_mole/unit.angstrom**2
m3t_openmm.add_harmonic_restraint_in_distances (system,
                                                atoms_pairs,
                                                harmonic_constant,
                                                system_positions = minimized_positions)

## Harmonic restraint distances

#atoms_sel = m3t.select(topology, 'resid 116 to 169 and name CA')
#atoms_pairs = [ [atoms_sel[0],ii] for ii in atoms_sel[1:] ]
#atoms_pairs.extend([ [atoms_sel[-1],ii] for ii in atoms_sel[1:-1] ])
#harmonic_constant = 50.0 * unit.kilocalories_per_mole/unit.angstrom**2
#m3t_openmm.add_harmonic_restraint_in_distances (system,
#                                                atoms_pairs,
#                                                harmonic_constant,
#                                                system_positions = minimized_positions)
#

# Pulling

atom_to_pull = m3t.select(topology, 'resid 0 name CA')[0]
vect_to_pull = np.array([0.0, -10.0, 0.0])
vect_to_pull = vect_to_pull/np.linalg.norm(vect_to_pull)
force = 500.0 * vect_to_pull * unit.kilocalories_per_mole/unit.angstrom
#m3t.utils.openmm.add_constant_pulling_force(system, atom_to_pull, force)

# Setting up the simulation parameters

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
temperature = 0*unit.kelvin
pressure    = None

friction   = 500.0/unit.picosecond
step_size  = 2.0*unit.femtoseconds
integrator = mm.LangevinIntegrator(temperature, friction, step_size)
integrator.setConstraintTolerance(0.00001)

simulation_time = 500000*unit.femtoseconds
saving_time = 1000*unit.femtoseconds
printout_time = 5000*unit.femtoseconds
simulation_steps = round(simulation_time/step_size)
saving_steps     = round(saving_time/step_size)
printout_steps     = round(printout_time/step_size)
num_steps_saved  = round(simulation_steps/saving_steps)

print('Simulation steps:', simulation_steps)
print('Saving steps:', saving_steps)
print('Saved steps:', num_steps_saved)

platform = mm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}

simulation = app.Simulation(topology, system, integrator, platform, properties)

simulation.context.setPositions(initial_positions)
simulation.context.setVelocitiesToTemperature(0*unit.kelvin)

#simulation.minimizeEnergy()

simulation.reporters.append(app.DCDReporter('trajectory.dcd', saving_steps))
simulation.reporters.append(app.StateDataReporter(stdout, printout_steps, step=True,
                            progress=True, remainingTime=True,
                            speed=True, totalSteps=simulation_steps, separator='\t'))

simulation.step(simulation_steps)


