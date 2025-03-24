#!/usr/bin/env python3
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Chemin complet vers le fichier SDF du robot')
    parser.add_argument('--entity_name', required=True, help='Nom de l\'entité')
    parser.add_argument('--x', default='0.0')
    parser.add_argument('--y', default='0.0')
    parser.add_argument('--z', default='0.0')
    parser.add_argument('--R', default='0.0')
    parser.add_argument('--P', default='0.0')
    parser.add_argument('--Y', default='0.0')
    args = parser.parse_args()

    try:
        with open(args.file, 'r') as f:
            sdf_content = f.read()
    except Exception as e:
        print("Erreur lors de la lecture du fichier SDF:", e)
        return

    # Appel du service create pour spawner le robot via ign service
    # La requête attend un message de type ignition.msgs.StringMsg avec un champ data contenant le XML SDF.
    cmd = [
        "ign", "service", "-s", "/world/Line_track/create",
        "--reqtype", "ignition.msgs.StringMsg",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "1000",
        "--req", f"data: '{sdf_content}'"
    ]
    print("Lancement de la commande:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        print("Robot spawné avec succès.")
    else:
        print("Echec du spawn du robot. Erreur:", result.stderr)

if __name__ == '__main__':
    main()

