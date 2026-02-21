import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from ase.io import read
from gpaw import GPAW, FermiDirac
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime

# Create output directory
os.makedirs("phonopy_outputs", exist_ok=True)

# Progress tracking
CHECKPOINT_FILE = "phonopy_outputs/checkpoint.pkl"

# Open progress log
progress_log = open("phonopy_outputs/progress_log.txt", "a")

def log(message):
    """Write to both screen and log file"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    progress_log.write(full_message + "\n")
    progress_log.flush()

def save_checkpoint(stage, data):
    """Save progress checkpoint"""
    checkpoint = {'stage': stage, 'data': data}
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)
    log(f"✓ Checkpoint saved: {stage}")

def load_checkpoint():
    """Load progress checkpoint"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return None

# ==================================================
# Check for existing checkpoint
# ==================================================
checkpoint = load_checkpoint()
if checkpoint:
    log("=" * 60)
    log(f"RESUMING FROM CHECKPOINT: {checkpoint['stage']}")
    log("=" * 60)
else:
    log("=" * 60)
    log("STARTING NEW PHONON CALCULATION")
    log("=" * 60)

# ==================================================
# 1) Read structure
# ==================================================
ase_atoms = read("Ag_relax.vasp")
log(f"Structure: {ase_atoms.get_chemical_formula()}, {len(ase_atoms)} atoms")

phonopy_atoms = PhonopyAtoms(
    symbols=ase_atoms.get_chemical_symbols(),
    cell=ase_atoms.cell,
    scaled_positions=ase_atoms.get_scaled_positions()
)

# ==================================================
# 2) Create Phonopy object
# ==================================================
phonon = Phonopy(
    phonopy_atoms,
    supercell_matrix=[[3, 0, 0],
                      [0, 3, 0],
                      [0, 0, 1]],
    primitive_matrix=None
)

log(f"Supercell: {len(phonon.supercell.symbols)} atoms")
phonon.generate_displacements(distance=0.01)
log(f"Number of displacements: {len(phonon.displacements)}")
log("")

supercells = phonon.supercells_with_displacements

# ==================================================
# 3) Calculate forces (RESUMABLE)
# ==================================================
if checkpoint and checkpoint['stage'] == 'forces_done':
    log("Loading saved forces from checkpoint...")
    forces_list = checkpoint['data']['forces_list']
    energies_list = checkpoint['data']['energies_list']
    log(f"✓ Loaded {len(forces_list)} force calculations")
else:
    log("=" * 60)
    log("CALCULATING FORCES")
    log("=" * 60)
    
    calc = GPAW(
        mode="lcao",
        xc="PBE",
        kpts=(4, 4, 1),
        occupations=FermiDirac(0.01),
        symmetry="off",
        txt="phonopy_outputs/gpaw_calculation.txt"
    )
    
    forces_list = []
    energies_list = []
    forces_csv = open("phonopy_outputs/forces_summary.csv", "w")
    forces_csv.write("Displacement_ID,Energy_eV,Max_Force_eV_per_A,Mean_Force_eV_per_A\n")
    
    for i, scell in enumerate(supercells):
        log(f"Displacement {i+1}/{len(supercells)}")
        
        from ase import Atoms
        ase_scell = Atoms(
            symbols=scell.symbols,
            scaled_positions=scell.scaled_positions,
            cell=scell.cell,
            pbc=True
        )
        
        ase_scell.calc = calc
        energy = ase_scell.get_potential_energy()
        forces = ase_scell.get_forces()
        
        forces_list.append(forces)
        energies_list.append(energy)
        
        max_force = np.max(np.abs(forces))
        mean_force = np.mean(np.abs(forces))
        
        log(f"  Energy: {energy:.6f} eV")
        log(f"  Max force: {max_force:.6f} eV/A")
        log("")
        
        forces_csv.write(f"{i},{energy:.8f},{max_force:.8f},{mean_force:.8f}\n")
        
        # Save individual force file
        force_file = f"phonopy_outputs/forces_displacement_{i:03d}.txt"
        with open(force_file, "w") as f:
            f.write(f"# Forces for displacement {i}\n")
            f.write(f"# Energy: {energy:.8f} eV\n")
            f.write(f"# Atom  Symbol  Fx  Fy  Fz  |F|\n")
            for atom_idx, (symbol, force) in enumerate(zip(scell.symbols, forces)):
                fmag = np.linalg.norm(force)
                f.write(f"{atom_idx:4d}  {symbol:2s}  {force[0]:12.6f}  {force[1]:12.6f}  {force[2]:12.6f}  {fmag:12.6f}\n")
    
    forces_csv.close()
    log("✓ All forces calculated")
    
    # Save checkpoint
    save_checkpoint('forces_done', {
        'forces_list': forces_list,
        'energies_list': energies_list
    })

# ==================================================
# 4) Build force constants
# ==================================================
log("")
log("=" * 60)
log("BUILDING FORCE CONSTANTS")
log("=" * 60)

phonon.forces = forces_list
phonon.produce_force_constants()
log("✓ Force constants calculated")

# ==================================================
# 5) Gamma frequencies
# ==================================================
log("")
log("=" * 60)
log("GAMMA-POINT FREQUENCIES")
log("=" * 60)

phonon.run_mesh([20, 20, 1])
mesh_dict = phonon.get_mesh_dict()
gamma_freqs = mesh_dict['frequencies'][0]

log("\nFrequencies at Gamma:")
log("Mode  Frequency(THz)  Frequency(cm^-1)  Type")
log("-" * 60)

gamma_csv = open("phonopy_outputs/gamma_frequencies.csv", "w")
gamma_csv.write("Mode,Frequency_THz,Frequency_cm_inv,Type\n")

for i, f in enumerate(gamma_freqs):
    cm_inv = f * 33.356
    mode_type = "acoustic" if i < 3 else "optical"
    log(f"{i:4d}  {f:14.6f}  {cm_inv:16.2f}  {mode_type}")
    gamma_csv.write(f"{i},{f:.8f},{cm_inv:.4f},{mode_type}\n")

gamma_csv.close()

if np.any(gamma_freqs < -0.01):
    log("\n⚠ WARNING: Imaginary frequencies detected")
else:
    log("\n✓ All frequencies positive")

# ==================================================
# 6) Band structure
# ==================================================
log("")
log("=" * 60)
log("CALCULATING BAND STRUCTURE")
log("=" * 60)

bands_dict = {
    'path': [
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0]],
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.0]],
        [[0.0, 0.5, 0.0], [0.0, 0.0, 0.0]]
    ],
    'labels': ['$\\Gamma$', 'X', 'M', 'Y', '$\\Gamma$'],
    'npoints': 51
}

phonon.run_band_structure(bands_dict['path'], bands_dict['npoints'])
bs_dict = phonon.get_band_structure_dict()
log("✓ Band structure calculated")

# ==================================================
# 7) Plot
# ==================================================
log("")
log("=" * 60)
log("PLOTTING")
log("=" * 60)

# Extract and convert data
distances_list = bs_dict['distances']
frequencies_list = bs_dict['frequencies']

# Convert to proper numpy arrays
all_distances = []
all_frequencies = []

for segment_idx in range(len(distances_list)):
    segment_distances = np.array(distances_list[segment_idx])
    segment_freqs = np.array(frequencies_list[segment_idx])
    all_distances.append(segment_distances)
    all_frequencies.append(segment_freqs)

# Concatenate all segments
distances_array = np.concatenate(all_distances)
frequencies_array = np.concatenate(all_frequencies)

# Convert THz to cm^-1
frequencies_cm = frequencies_array * 33.356

log(f"Data shape: {frequencies_cm.shape}")

# Save to CSV
bands_csv = open("phonopy_outputs/phonon_bands.csv", "w")
bands_csv.write("k_distance")
for i in range(frequencies_cm.shape[1]):
    bands_csv.write(f",band_{i}")
bands_csv.write("\n")

for i in range(len(distances_array)):
    bands_csv.write(f"{distances_array[i]:.6f}")
    for band_idx in range(frequencies_cm.shape[1]):
        bands_csv.write(f",{frequencies_cm[i][band_idx]:.4f}")
    bands_csv.write("\n")

bands_csv.close()
log("✓ Data saved to CSV")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

for band_idx in range(frequencies_cm.shape[1]):
    ax.plot(distances_array, frequencies_cm[:, band_idx], 'b-', linewidth=1.5)

ax.set_xlabel('Wave vector', fontsize=12)
ax.set_ylabel('Frequency (cm$^{-1}$)', fontsize=12)
ax.set_title(f'Phonon Band Structure - {ase_atoms.get_chemical_formula()}', fontsize=13)
ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_ylim(bottom=-200, top=1000)
ax.grid(True, alpha=0.3, axis='y')

# High-symmetry point markers
segment_positions = [0]
for segment in all_distances:
    segment_positions.append(segment[-1])

for pos, label in zip(segment_positions, bands_dict['labels']):
    ax.axvline(pos, color='k', linewidth=0.5, alpha=0.5)
ax.set_xticks(segment_positions)
ax.set_xticklabels(bands_dict['labels'])

plt.tight_layout()
plt.savefig('phonopy_outputs/phonon_bandstructure.png', dpi=300)
log("✓ Plot saved")
plt.show()

# Save phonopy params
phonon.save('phonopy_outputs/phonopy_params.yaml')

log("")
log("=" * 60)
log("COMPLETE")
log("=" * 60)

progress_log.close()
print("\n✓ All outputs in phonopy_outputs/")
print("✓ To restart from scratch: rm phonopy_outputs/checkpoint.pkl")
