from pyrosetta import init, pose_from_sequence

# Initialize PyRosetta
init()

# Create a pose from sequence
seq_pose = pose_from_sequence('GCTAGCTACGATGCA')

# Dump the pose to a PDB file
seq_pose.dump_pdb('./seq1.pdb')
