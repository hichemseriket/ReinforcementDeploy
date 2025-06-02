# streamlit_app.py
# Pour lancer en mode serveur local accessible sur le réseau LAN :
# 1) Identifiez l'adresse IP de votre machine (ex: `ipconfig` sous Windows ou `ifconfig` sous Linux/macOS`).
# 2) Lancez Streamlit :
#      streamlit run src/streamlitFred.py \
#          --server.address 0.0.0.0 \
#          --server.port 8501 \
#          --server.headless true
# 3) Depuis un autre poste du même réseau, ouvrez :
#      http://<IP_DE_VOTRE_MACHINE>:8501
# 4) Vérifiez que le pare-feu autorise les connexions entrantes sur le port 8501.


# src/streamlitFred.py

import os
import sys
import time
import random
import tempfile
import pickle

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go

# ────────────────────────────────────────────────────────────────
# 1) On ajoute la racine src/ au PATH pour retrouver rl_infra_private et plotting/
HERE     = os.path.dirname(__file__)                             # …/renforcementPolyvoxels/src
BASE_DIR = os.path.abspath(os.path.join(HERE, os.pardir))        # …/renforcementPolyvoxels

# ────────────────────────────────────────────────────────────────

# 2) IMPORTS via votre package privé
from rl_infra_private.agent.QLearningAgent         import QLearningAgent
from rl_infra_private.environment.VoxelEnvironment import VoxelEnvironment

# 3) IMPORT du module de plotting (reste dans src/plotting/)
from plotting.plotting import plot_voxel_environment


def get_csv_path(rel_path: str) -> str:
    """
    rel_path : chemin relatif **depuis** dataInput/,
               par ex. "DATAPointsCSV/cube_points_6x6x6.csv"
    """
    data_dir = os.path.join(BASE_DIR, "dataInput")
    p = os.path.join(data_dir, rel_path)
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"CSV introuvable : {p}")



MODEL_DIR = os.path.join(HERE, "models")


def run_inference(env, agent, max_steps=100000, target_dof=3):
    state = env.reset()
    rewards, step_times, blocks, dofs = [], [], [], []
    steps = 0
    t0_global = time.time()

    while steps < max_steps:
        _, adj = env.get_adjacent_entities()
        if not adj:
            break
        t0 = time.time()
        if random.random() < agent.exploration_rate:
            action = random.choice(adj)
        else:
            action = max(adj, key=lambda a: agent.get_q_value(state, a))
        state, r, done = env.step(action)
        step_times.append(time.time() - t0)
        rewards.append(r)
        blocks.append(len(env.polyvoxels))
        dofs.append(env.current_dof)
        steps += 1
        # Mise à jour du progress bar
        progress = int(steps / int(max_steps) * 100)
        progress_text.text(f"Progression : {progress}% ({steps}/{int(max_steps)})")
        progress_bar.progress(progress)
        if done or env.current_dof <= target_dof:
            break

    return {
        'rewards': rewards,
        'step_times': step_times,
        'blocks': blocks,
        'dofs': dofs,
        'steps': steps,
        'total_time': time.time() - t0_global,
        'final_blocks': blocks[-1] if blocks else 0,
        'final_dof': dofs[-1] if dofs else None
    }


# ────────────────────────────────────────────────────────────────
# Interface Streamlit
st.set_page_config(page_title="4D Interlocking Inference", layout='wide')
st.title("Prototype d'inférence 4D Interlocking")

# Sidebar: exemples prédéfinis
st.sidebar.header("Fichiers exemples")
EX_CSV = {
    "Cube 6×6×6": get_csv_path("DATAPointsCSV/cube_points_6x6x6.csv"),
    "Cube 8×8×8": get_csv_path("DATAPointsCSV/cube_points_8x8x8.csv"),
    "Sphere 1000 pts": get_csv_path("DATAPointsCSV/sphere_1000_points.csv"),
    "Lion": get_csv_path("DATAPointsCSV/lionDayl.csv"),
    "Lapin 1700 pts": get_csv_path("DATAPointsCSV/bunny1700.csv"),
    "Poutre Article": get_csv_path("DATAPointsCSV/Toyota25x8x4Article.csv"),
    "Lapin 2500 pts": get_csv_path("DATAPointsCSV/bunny2500Point.csv"),
    "Cube 16x16x16": get_csv_path("DATAPointsCSV/cube_points_16x16x6.csv"),
    "Arche": get_csv_path("DATAPointsCSV/newSphereTrancheVide.csv"),
    "Tranche Sphere": get_csv_path("DATAPointsCSV/newSphereTranche.csv"),
}

st.sidebar.header("Modèles exemples")
EX_MODEL = {
    "Q-Cube":   os.path.join(MODEL_DIR, "qlearning_cube6x6x6.pkl"),
    "Q-BigCube":   os.path.join(MODEL_DIR, "qlearning_cube_points_8x8x8.pkl"),
    "Q-Sphere": os.path.join(MODEL_DIR, "qlearning_sphere.pkl"),
    "Q-Poutre": os.path.join(MODEL_DIR, "qlearning_Poutre.pkl"),
    "Q-Bunny":  os.path.join(MODEL_DIR, "qlearning_Bunny.pkl"),
    "Q-BigBunny":  os.path.join(MODEL_DIR, "qlearning_1500V_Bunny.pkl"),
    "Q-Lion":   os.path.join(MODEL_DIR, "qlearning_lion.pkl"),
    "Q-Demo":   os.path.join(MODEL_DIR, "qlearning_demo.pkl"),
    "Q-PlotDemo":   os.path.join(MODEL_DIR, "qlearning_demo_Plot.pkl"),
}

choice_csv   = st.sidebar.selectbox("CSV exemple",   list(EX_CSV.keys()),   index=0)
choice_model = st.sidebar.selectbox("Modèle exemple", list(EX_MODEL.keys()), index=0)

# Téléversement manuel
st.sidebar.markdown("**Ou téléversez vos propres fichiers**")
model_file = st.sidebar.file_uploader("Modèle Q-Learning (.pkl)", type=["pkl"])
csv_file   = st.sidebar.file_uploader("Points (CSV)",         type=["csv"])

# Hyperparamètres
st.sidebar.header("Paramètres")
seed          = st.sidebar.number_input("Seed aléatoire",      min_value=0,   value=42)
epsilon       = st.sidebar.number_input("Epsilon (exploration)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
max_steps     = st.sidebar.number_input("Max steps",           min_value=1,   value=5000)
target_dof    = st.sidebar.number_input("DOF cible",           min_value=0,   value=3)
max_poly_size = st.sidebar.number_input("Max polyvoxel size",  min_value=1,   value=60)

# Définition des chemins finaux
if model_file:
    model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl").name
    with open(model_path, "wb") as f:
        f.write(model_file.getvalue())
else:
    model_path = EX_MODEL[choice_model]

if csv_file:
    csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    with open(csv_path, "wb") as f:
        f.write(csv_file.getvalue())
else:
    csv_path = EX_CSV[choice_csv]

# Aperçu 3D initial
st.write(f"**Fichier chargé :** `{os.path.basename(csv_path)}`")
try:
    df_in = pd.read_csv(csv_path, sep=None, engine="python", decimal=",")
except Exception:
    df_in = pd.read_csv(csv_path, sep=";", decimal=",")

if not df_in.empty and df_in.select_dtypes(include="number").shape[1] >= 3:
    num_cols = df_in.select_dtypes(include="number").columns.tolist()[:3]
    pts = df_in[num_cols].values
    traces_in = []
    for x0, y0, z0 in pts:
        verts = np.array([
            [x0,   y0,   z0],
            [x0+1, y0,   z0],
            [x0+1, y0+1, z0],
            [x0,   y0+1, z0],
            [x0,   y0,   z0+1],
            [x0+1, y0,   z0+1],
            [x0+1, y0+1, z0+1],
            [x0,   y0+1, z0+1],
        ])
        I, J, K = (
            [0,0,0,1,1,2,3,4,5,6,4,5],
            [1,3,4,2,5,3,7,5,6,7,7,6],
            [3,4,5,3,6,7,4,7,7,5,6,4]
        )
        traces_in.append(go.Mesh3d(
            x=verts[:,0], y=verts[:,1], z=verts[:,2],
            i=I, j=J, k=K,
            opacity=0.2, color='gray', showscale=False
        ))
    fig_in = go.Figure(data=traces_in)
    fig_in.update_layout(scene=dict(aspectmode="data"), title="Aperçu 3D des points d'entrée")
    st.plotly_chart(fig_in, use_container_width=True)

# Lancer l'inférence
if st.button("Lancer l'inférence"):
    random.seed(seed)
    np.random.seed(seed)

    env   = VoxelEnvironment(max_polyvoxel_size=int(max_poly_size))
    env.load_points_from_csv(csv_path)

    with open(model_path, "rb") as f:
        q_table_saved = pickle.load(f)

    agent = QLearningAgent(
        state_size=None, action_size=None,
        grid_size_x=env.grid_size_x, grid_size_y=env.grid_size_y, grid_size_z=env.grid_size_z,
        learning_rate=0.1, discount_rate=0.9,
        exploration_rate=epsilon, exploration_decay=1.0
    )
    agent.q_table = q_table_saved
    # Progression
    progress_text = st.empty()
    progress_bar = st.progress(0)

    result = run_inference(env, agent, max_steps=int(max_steps), target_dof=int(target_dof))
    progress_bar.empty()
    progress_text.empty()
    # Affichage des résultats
    st.subheader("Résultats")
    cols = st.columns(4)
    cols[0].metric("Étapes",          result['steps'])
    cols[1].metric("Temps total (s)", f"{result['total_time']:.2f}")
    cols[2].metric("Polyvoxels finaux", result['final_blocks'])
    cols[3].metric("DOF final",        result['final_dof'])

    # Préparez le DataFrame pour le téléchargement
    df_out = pd.DataFrame({
        'step':    list(range(1, result['steps']+1)),
        'blocks':  result['blocks'],
        'dofs':    result['dofs'],
        'rewards': result['rewards']
    })
    st.download_button(
        "Télécharger CSV détaillé",
        df_out.to_csv(index=False).encode("utf-8"),
        "distribution.csv",
        "text/csv"
    )

    # Historique
    st.subheader("Historique Blocks & DOF")
    st.line_chart(pd.DataFrame({'Blocks': result['blocks'], 'DOF': result['dofs']}))

    st.subheader("Distribution finale interactive (polyvoxels emboîtés)")

    from plotly.colors import qualitative

    palette = qualitative.Plotly
    traces = []

    for idx, poly in enumerate(env.polyvoxels):
        color = palette[idx % len(palette)]
        X, Y, Z = [], [], []
        I, J, K = [], [], []
        vert_offset = 0

        # Concatène tous les voxels dans un seul mesh
        for vox in poly.voxels:
            x0, y0, z0 = vox.x, vox.y, vox.z
            verts = np.array([
                [x0, y0, z0],
                [x0 + 1, y0, z0],
                [x0 + 1, y0 + 1, z0],
                [x0, y0 + 1, z0],
                [x0, y0, z0 + 1],
                [x0 + 1, y0, z0 + 1],
                [x0 + 1, y0 + 1, z0 + 1],
                [x0, y0 + 1, z0 + 1]
            ])
            X.extend(verts[:, 0]);
            Y.extend(verts[:, 1]);
            Z.extend(verts[:, 2])

            faces = [
                (0, 1, 2), (0, 2, 3),
                (4, 5, 6), (4, 6, 7),
                (0, 1, 5), (0, 5, 4),
                (2, 3, 7), (2, 7, 6),
                (1, 2, 6), (1, 6, 5),
                (0, 3, 7), (0, 7, 4)
            ]
            for a, b, c in faces:
                I.append(a + vert_offset)
                J.append(b + vert_offset)
                K.append(c + vert_offset)

            vert_offset += 8

        traces.append(
            go.Mesh3d(
                x=X, y=Y, z=Z,
                i=I, j=J, k=K,
                color=color,
                opacity=0.4,
                name=f"Polyvoxel {idx + 1}",
                showlegend=True
            )
        )

    fig_dist = go.Figure(data=traces)
    fig_dist.update_layout(
        title="Distribution finale des polyvoxels",
        scene=dict(aspectmode="data"),
        legend_title_text="Polyvoxels",
        showlegend=True
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.success("Inférence terminée !")
