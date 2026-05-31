import os
import tempfile
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as sp_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import streamlit as st
import ezc3d

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(page_title="GPBIO - Biomecânica Clínica", layout="wide", page_icon="🚶")

# =============================================================================
# ENGINE MATEMÁTICA E SEGMENTAR
# =============================================================================
class ProcessadorCinematico:
    def __init__(self, caminho_arquivo, nome_original, grupo="Geral", df_antropo=None):
        self.caminho = caminho_arquivo
        self.nome_arq = nome_original
        self.grupo = grupo
        self.valido = False
        self.erro_msg = ""
        
        nome_limpo = nome_original.lower().replace('.c3d', '')
        self.id_paciente = nome_limpo.split('_')[0].upper().strip()

        try:
            self.c3d = ezc3d.c3d(caminho_arquivo)
            self.labels = [l.strip() for l in self.c3d['parameters']['POINT']['LABELS']['value']]
            self.mapa = {lbl: i for i, lbl in enumerate(self.labels)}
            self.freq = self.c3d['parameters']['POINT']['RATE']['value'][0]
            self.dados_raw = self.c3d['data']['points'][:3, :, :]
            self.n_frames = self.dados_raw.shape[2]

            self.dados = self._filtrar()
            
            rias = self._get('RIAS', slice(None))
            if rias is not None and rias.shape[1] > 0:
                prog = rias[0, -1] - rias[0, 0]
                self.dir_x = 1 if prog >= 0 else -1
            else:
                self.dir_x = 1
            
            self.segmentos_df = self._calcular_angulos_segmentares() 
            self.angulos_df = self._calcular_angulos() 
            
            self.velocidade_media = self._calcular_velocidade_sacrum()
            self.eventos = self.detectar_eventos_zeni()
            self.fases_marcha = self._calcular_fases_marcha()
            self.foot_clearance = self._calcular_foot_clearance()
            self.comprimento_passo = self._calcular_comprimento_passo()

            self.passo_norm = {'D': np.nan, 'E': np.nan}
            if df_antropo is not None:
                match = df_antropo[df_antropo['ID'] == self.id_paciente]
                if not match.empty:
                    altura_m = float(match['ALTURA'].values[0])
                    if altura_m > 3.0: 
                        altura_m = altura_m / 100.0
                    if altura_m > 0:
                        val_d = self.comprimento_passo.get('D', np.nan)
                        val_e = self.comprimento_passo.get('E', np.nan)
                        if not np.isnan(val_d) and val_d > 0: 
                            self.passo_norm['D'] = ((val_d / 1000.0) / altura_m) * 100.0
                        if not np.isnan(val_e) and val_e > 0: 
                            self.passo_norm['E'] = ((val_e / 1000.0) / altura_m) * 100.0
            
            self.coord_vetorial = self._calcular_coordenacao_vetorial()
            self.valido = True
        except Exception as e:
            self.erro_msg = str(e)
            self.valido = False
            
    def _get(self, nome, f):
        idx = self.mapa.get(nome)
        if idx is not None:
            v = self.dados[:, idx, f]
            if not np.isnan(v).any(): 
                return v
        return None

    def _mid(self, n1, n2, f):
        p1, p2 = self._get(n1, f), self._get(n2, f)
        if p1 is not None and p2 is not None: 
            return (p1+p2)/2
        return None

    def _filtrar(self):
        d = self.dados_raw.copy()
        d[d==0.0] = np.nan
        nyq = 0.5 * self.freq
        b, a = signal.butter(4, 6.0/nyq, btype='low')
        out = np.zeros_like(d) * np.nan
        for m in range(d.shape[1]):
            for ax in range(3):
                sinal_d = d[ax, m, :]
                if np.isnan(sinal_d).all(): 
                    continue
                s_temp = pd.Series(sinal_d).interpolate(limit_direction='both').bfill().ffill()
                try:
                    out[ax, m, :] = signal.filtfilt(b, a, s_temp.to_numpy())
                except Exception: 
                    out[ax, m, :] = s_temp
        return out

    def _ang_sagital_vert(self, p_prox, p_dist):
        if p_prox is None or p_dist is None: return np.nan
        dx = (p_dist[0] - p_prox[0]) * self.dir_x
        dz = p_dist[2] - p_prox[2] 
        return np.degrees(np.arctan2(dx, -dz))

    def _ang_sagital_horiz(self, p_prox, p_dist):
        if p_prox is None or p_dist is None: return np.nan
        dx = (p_dist[0] - p_prox[0]) * self.dir_x
        dz = p_dist[2] - p_prox[2]
        return np.degrees(np.arctan2(dz, dx))

    def _calcular_angulos_segmentares(self):
        res = {k: [] for k in ['Coxa_D','Perna_D','Pe_D','Coxa_E','Perna_E','Pe_E']}
        for f in range(self.n_frames):
            for lado, l in [('D', 'R'), ('E', 'L')]:
                h = self._get(f'{l}IAS',f); k = self._mid(f'{l}LE', f'{l}ME',f); a = self._mid(f'{l}ML', f'{l}MM',f)
                p = self._mid(f'{l}FT1', f'{l}FT5',f); cal = self._get(f'{l}CAL',f)
                
                res[f'Coxa_{lado}'].append(self._ang_sagital_vert(h, k))
                res[f'Perna_{lado}'].append(self._ang_sagital_vert(k, a))
                res[f'Pe_{lado}'].append(self._ang_sagital_horiz(cal, p))
        return pd.DataFrame(res)

    def _calcular_angulos(self):
        res = {k: [] for k in ['Quad_D','Joel_D','Torn_D','Quad_E','Joel_E','Torn_E']}
        for f in range(self.n_frames):
            for lado in ['D', 'E']:
                coxa = self.segmentos_df[f'Coxa_{lado}'][f]
                perna = self.segmentos_df[f'Perna_{lado}'][f]
                pe = self.segmentos_df[f'Pe_{lado}'][f]
                
                res[f'Quad_{lado}'].append(coxa) 
                res[f'Joel_{lado}'].append(coxa - perna if not (np.isnan(coxa) or np.isnan(perna)) else np.nan)
                res[f'Torn_{lado}'].append(perna + pe if not (np.isnan(pe) or np.isnan(perna)) else np.nan)
        return pd.DataFrame(res)

    def detectar_eventos_zeni(self):
        eventos = {'D': {'HS': [], 'TO': []}, 'E': {'HS': [], 'TO': []}}
        rias_data, lias_data = self._get('RIAS', slice(None)), self._get('LIAS', slice(None))
        if rias_data is None or lias_data is None: return eventos 
            
        pelvis_x = (rias_data[0] + lias_data[0]) / 2
        dist_frames = int(self.freq * 0.35)
        
        for lado, cal_label, toe_label in [('D','RCAL','RFT1'), ('E','LCAL','LFT1')]:
            cal_x_data, toe_x_data = self._get(cal_label, slice(None)), self._get(toe_label, slice(None))
            if cal_x_data is None or toe_x_data is None: continue
            
            curve_hs = (cal_x_data[0] - pelvis_x) * self.dir_x
            curve_to = (toe_x_data[0] - pelvis_x) * self.dir_x
            
            if np.nanmean(curve_hs) > 0:
                picos_hs, _ = signal.find_peaks(-curve_hs, distance=dist_frames)
                vales_to, _ = signal.find_peaks(curve_to, distance=dist_frames)
            else:
                picos_hs, _ = signal.find_peaks(curve_hs, distance=dist_frames)
                vales_to, _ = signal.find_peaks(-curve_to, distance=dist_frames)
                
            eventos[lado]['HS'], eventos[lado]['TO'] = sorted(picos_hs), sorted(vales_to)
        return eventos

    def obter_stats(self):
        if not self.valido: return None
        return {col: {'min': self.angulos_df[col].min(), 'max': self.angulos_df[col].max()} for col in self.angulos_df.columns}

    def _calcular_velocidade_sacrum(self):
        rips, lips = self._get('RIPS', slice(None)), self._get('LIPS', slice(None))
        if rips is None or lips is None: return np.nan
        sacrum = (rips + lips) / 2
        return np.nanmean(np.linalg.norm(np.diff(sacrum, axis=1), axis=0) * self.freq / 1000.0)

    def _calcular_fases_marcha(self):
        res = {'D': {'Apoio': np.nan, 'Balanco': np.nan}, 'E': {'Apoio': np.nan, 'Balanco': np.nan}}
        for lado in ['D', 'E']:
            hss, tos = self.eventos[lado]['HS'], self.eventos[lado]['TO']
            if len(hss) < 2 or not tos: continue
            ciclos_apoio = []
            for i in range(len(hss) - 1):
                to_valido = [t for t in tos if hss[i] < t < hss[i+1]]
                if to_valido:
                    pct = ((to_valido[0] - hss[i]) / (hss[i+1] - hss[i])) * 100
                    if pct < 45.0: pct = 100.0 - pct
                    ciclos_apoio.append(pct)
            if ciclos_apoio: 
                res[lado]['Apoio'] = np.mean(ciclos_apoio)
                res[lado]['Balanco'] = 100.0 - np.mean(ciclos_apoio)
        return res

    def _calcular_foot_clearance(self):
        res = {'D': np.nan, 'E': np.nan}
        for lado, pref in [('D', 'R'), ('E', 'L')]:
            ft1 = self._get(f'{pref}FT1', slice(None))
            if ft1 is None: continue
            hss, tos = self.eventos[lado]['HS'], self.eventos[lado]['TO']
            alturas = [np.max(ft1[2, to_f:min([h for h in hss if h > to_f] or [to_f])]) for to_f in tos if [h for h in hss if h > to_f]]
            if alturas: res[lado] = np.mean(alturas)
        return res

    def _calcular_comprimento_passo(self):
        res = {'D': np.nan, 'E': np.nan}
        rcal, lcal = self._get('RCAL', slice(None)), self._get('LCAL', slice(None))
        rft1, lft1 = self._get('RFT1', slice(None)), self._get('LFT1', slice(None))
        if any(m is None for m in [rcal, lcal, rft1, lft1]): return res
        passos_d = [np.linalg.norm(rcal[0:2, hs] - lft1[0:2, hs]) for hs in self.eventos['D']['HS'] if hs < rcal.shape[1] and hs < lft1.shape[1]]
        passos_e = [np.linalg.norm(lcal[0:2, hs] - rft1[0:2, hs]) for hs in self.eventos['E']['HS'] if hs < lcal.shape[1] and hs < rft1.shape[1]]
        if passos_d: res['D'] = np.mean(passos_d)
        if passos_e: res['E'] = np.mean(passos_e)
        return res

    def extrair_ciclos_normalizados(self, vetor_dados, eventos_hs, pontos=101):
        ciclos = []
        if len(eventos_hs) < 2: return []
        for i in range(len(eventos_hs) - 1):
            if eventos_hs[i+1] > len(vetor_dados): continue
            ciclo_bruto = vetor_dados[eventos_hs[i]:eventos_hs[i+1]]
            ciclos.append(np.interp(np.linspace(0, len(ciclo_bruto)-1, pontos), np.arange(len(ciclo_bruto)), ciclo_bruto))
        return ciclos

    def _calcular_coordenacao_vetorial(self):
        res = {}
        self.coord_vetorial_series = {} 
        
        for lado in ['D', 'E']:
            hss = self.eventos[lado]['HS']
            if len(hss) < 2: continue
            
            pares_articulares = [
                (f'Quad_Joel_{lado}', f'Quad_{lado}', f'Joel_{lado}', self.angulos_df),
                (f'Joel_Torn_{lado}', f'Joel_{lado}', f'Torn_{lado}', self.angulos_df)
            ]
            
            pares_segmentares = [
                (f'Coxa_Perna_{lado}', f'Coxa_{lado}', f'Perna_{lado}', self.segmentos_df),
                (f'Perna_Pe_{lado}', f'Perna_{lado}', f'Pe_{lado}', self.segmentos_df)
            ]
            
            for nome_par, col_prox, col_dist, df_ref in pares_articulares + pares_segmentares:
                c_prox = self.extrair_ciclos_normalizados(df_ref[col_prox].values, hss)
                c_dist = self.extrair_ciclos_normalizados(df_ref[col_dist].values, hss)
                if not c_prox or not c_dist: continue
                
                res[nome_par] = {'Proximal': np.nan, 'Distal': np.nan, 'EmFase': np.nan, 'AntiFase': np.nan}
                freqs = {'Proximal': [], 'Distal': [], 'EmFase': [], 'AntiFase': []}
                
                for cp, cd in zip(c_prox, c_dist):
                    # Fórmula Exata de Vector Coding
                    angulos = np.mod(np.degrees(np.arctan2(np.diff(cd), np.diff(cp))), 360)
                    counts = {'Proximal': 0, 'Distal': 0, 'EmFase': 0, 'AntiFase': 0}
                    for a in angulos:
                        if (0 <= a < 22.5) or (337.5 <= a <= 360) or (157.5 <= a < 202.5): counts['Proximal'] += 1
                        elif (22.5 <= a < 67.5) or (202.5 <= a < 247.5): counts['EmFase'] += 1
                        elif (67.5 <= a < 112.5) or (247.5 <= a < 292.5): counts['Distal'] += 1
                        else: counts['AntiFase'] += 1
                    for k in counts: freqs[k].append((counts[k] / len(angulos)) * 100)
                
                for k in freqs: res[nome_par][k] = np.mean(freqs[k])
                
                c_prox_m = np.mean(c_prox, axis=0)
                c_dist_m = np.mean(c_dist, axis=0)
                ang_m = np.mod(np.degrees(np.arctan2(np.diff(c_dist_m), np.diff(c_prox_m))), 360)
                fatia_media = []
                for a in ang_m:
                    if (0 <= a < 22.5) or (337.5 <= a <= 360) or (157.5 <= a < 202.5): fatia_media.append('Proximal')
                    elif (22.5 <= a < 67.5) or (202.5 <= a < 247.5): fatia_media.append('EmFase')
                    elif (67.5 <= a < 112.5) or (247.5 <= a < 292.5): fatia_media.append('Distal')
                    else: fatia_media.append('AntiFase')
                
                self.coord_vetorial_series[nome_par] = fatia_media
        return res

# =============================================================================
# MÓDULO VISUAL (GIFs) 
# =============================================================================
class GeradorVisual:
    def __init__(self, processador, nome_original):
        self.proc = processador
        self.nome_arq = nome_original
        self.box = {'x': (-1000, 1000), 'y': (-1000, 1000), 'z': (0, 2000)}

    def montar_frame(self, f):
        s = {}
        get = lambda n: self.proc._get(n, f)
        mid = lambda n1, n2: self.proc._mid(n1, n2, f)
        rias, lias = get('RIAS'), get('LIAS')
        rips, lips = get('RIPS'), get('LIPS')
        rict, lict = get('RICT'), get('LICT')
        
        if rias is not None and lias is not None: s['P_F']=[rias,lias]
        if rips is not None and lips is not None: s['P_B']=[rips,lips]
        if rias is not None and rict is not None: s['PR1']=[rias,rict]
        if rips is not None and rict is not None: s['PR2']=[rips,rict] 
        if lias is not None and lict is not None: s['PL1']=[lias,lict]
        if lips is not None and lict is not None: s['PL2']=[lips,lict] 
        
        kd, ke = mid('RLE','RME'), mid('LLE','LME')
        td, te = mid('RML','RMM'), mid('LML','LMM')
        if rias is not None and kd is not None: s['CX_D']=[rias,kd]
        if lias is not None and ke is not None: s['CX_E']=[lias,ke]
        if kd is not None and td is not None: s['PN_D']=[kd,td]
        if ke is not None and te is not None: s['PN_E']=[ke,te]
            
        for l, cal, t1, t5, ank in [('D', get('RCAL'), get('RFT1'), get('RFT5'), td),
                                    ('E', get('LCAL'), get('LFT1'), get('LFT5'), te)]:
            if cal is not None and t1 is not None: s[f'P{l}1']=[cal,t1]
            if cal is not None and t5 is not None: s[f'P{l}2']=[cal,t5]
            if t1 is not None and t5 is not None: s[f'P{l}3']=[t1,t5]
            if ank is not None and cal is not None: s[f'P{l}L']=[ank,cal]
        return s

    def _desenhar_fundo_bussola(self, ax_c, titulo):
        ax_c.set_xlim(-1.2, 1.2); ax_c.set_ylim(-1.2, 1.2)
        ax_c.axis('off'); ax_c.set_aspect('equal')
        ax_c.text(0, 1.35, titulo, ha='center', va='center', fontsize=9, fontweight='bold')
        categorias = [((0, 22.5), '#e74c3c'), ((337.5, 360), '#e74c3c'), ((157.5, 202.5), '#e74c3c'),
                      ((22.5, 67.5), '#2ecc71'), ((202.5, 247.5), '#2ecc71'),
                      ((67.5, 112.5), '#3498db'), ((247.5, 292.5), '#3498db'),
                      ((112.5, 157.5), '#f1c40f'), ((292.5, 337.5), '#f1c40f')]
        for (t1, t2), cor in categorias: 
            ax_c.add_patch(mpatches.Wedge((0,0), 1.0, t1, t2, facecolor=cor, alpha=0.35, edgecolor='white', lw=1))
        ax_c.plot([0], [0], marker='o', color='black', markersize=4)
        ptr, = ax_c.plot([], [], color='black', lw=2.5)
        return ptr

    def _classificar_angulo(self, angulo):
        if np.isnan(angulo): return "-", "gray"
        a = angulo % 360
        if (0 <= a < 22.5) or (337.5 <= a <= 360) or (157.5 <= a < 202.5): return "PROXIMAL", '#e74c3c'
        elif (22.5 <= a < 67.5) or (202.5 <= a < 247.5): return "EM FASE", '#2ecc71'
        elif (67.5 <= a < 112.5) or (247.5 <= a < 292.5): return "DISTAL", '#3498db'
        else: return "ANTI-FASE", '#f1c40f'

    def salvar(self, caminho_final, step=3, fps_anim=20):
        fig = plt.figure(figsize=(16, 9))
        
        ax_comp_qj_d = fig.add_axes([0.01, 0.65, 0.15, 0.25]); ax_comp_jt_d = fig.add_axes([0.01, 0.38, 0.15, 0.25])
        ax_comp_qj_e = fig.add_axes([0.16, 0.65, 0.15, 0.25]); ax_comp_jt_e = fig.add_axes([0.16, 0.38, 0.15, 0.25])

        ptr_qjd = self._desenhar_fundo_bussola(ax_comp_qj_d, "Coxa-Perna (DIR)"); ptr_jtd = self._desenhar_fundo_bussola(ax_comp_jt_d, "Perna-Pé (DIR)")
        ptr_qje = self._desenhar_fundo_bussola(ax_comp_qj_e, "Coxa-Perna (ESQ)"); ptr_jte = self._desenhar_fundo_bussola(ax_comp_jt_e, "Perna-Pé (ESQ)")

        ax_stats_left = fig.add_axes([0.01, 0.02, 0.30, 0.32]); ax_stats_left.axis('off')
        
        ax = fig.add_axes([0.32, 0.20, 0.44, 0.75], projection='3d')
        ax.set_xlim(self.box['x']); ax.set_ylim(self.box['y']); ax.set_zlim(self.box['z']); ax.view_init(elev=20, azim=135)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        titulo_main = ax.set_title(self.nome_arq, fontsize=12, pad=20)

        ax_banner = fig.add_axes([0.33, 0.02, 0.43, 0.16]); ax_banner.axis('off')
        ax_txt = fig.add_axes([0.78, 0.05, 0.21, 0.90]); ax_txt.axis('off')

        stats_ang = self.proc.obter_stats(); coord_norm = self.proc.coord_vetorial

        ax_stats_left.text(0.5, 1.0, "FREQUÊNCIA NO CICLO DA MARCHA (0-100%)", ha='center', va='top', fontweight='bold', fontsize=10)
        def format_f(c): return f" Proximal : {c.get('Proximal',0):>3.0f}%\n Em Fase  : {c.get('EmFase',0):>3.0f}%\n Distal   : {c.get('Distal',0):>3.0f}%\n Anti-Fase: {c.get('AntiFase',0):>3.0f}%"
        
        col_dir = ">> COXA-PERNA (DIR)\n" + format_f(coord_norm.get('Coxa_Perna_D', {})) + "\n\n>> PERNA-PÉ (DIR)\n" + format_f(coord_norm.get('Perna_Pe_D', {}))
        col_esq = ">> COXA-PERNA (ESQ)\n" + format_f(coord_norm.get('Coxa_Perna_E', {})) + "\n\n>> PERNA-PÉ (ESQ)\n" + format_f(coord_norm.get('Perna_Pe_E', {}))

        ax_stats_left.text(0.00, 0.85, col_dir, va='top', fontsize=9, family='monospace'); ax_stats_left.text(0.55, 0.85, col_esq, va='top', fontsize=9, family='monospace')

        ax_banner.text(0.5, 0.90, "COORDENAÇÃO SEGMENTAR EM TEMPO REAL", ha='center', va='top', fontweight='bold', fontsize=11)
        ax_banner.text(0.00, 0.50, "Coxa-Perna (DIR):", fontweight='bold', fontsize=10); ax_banner.text(0.00, 0.15, "Perna-Pé (DIR):", fontweight='bold', fontsize=10)
        txt_qj_d = ax_banner.text(0.24, 0.50, "-", fontweight='bold', fontsize=10); txt_jt_d = ax_banner.text(0.24, 0.15, "-", fontweight='bold', fontsize=10)
        ax_banner.text(0.53, 0.50, "Coxa-Perna (ESQ):", fontweight='bold', fontsize=10); ax_banner.text(0.53, 0.15, "Perna-Pé (ESQ):", fontweight='bold', fontsize=10)
        txt_qj_e = ax_banner.text(0.77, 0.50, "-", fontweight='bold', fontsize=10); txt_jt_e = ax_banner.text(0.77, 0.15, "-", fontweight='bold', fontsize=10)

        t_dynamic = ax_txt.text(0.05, 0.95, "", va='top', fontsize=10, family='monospace')
        linhas = {}

        def update(i):
            seg = self.montar_frame(i)
            for k in list(linhas):
                if k not in seg: linhas[k].remove(); del linhas[k]
            for n, (p1, p2) in seg.items():
                c = 'red' if 'D' in n or 'R' in n else 'blue'
                if 'P_' in n or 'PL' in n or 'PR' in n: c = 'black'
                if n in linhas:
                    linhas[n].set_data([p1[0],p2[0]],[p1[1],p2[1]]); linhas[n].set_3d_properties([p1[2],p2[2]])
                else: 
                    linhas[n], = ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]], c=c, lw=1.2)

            row = self.proc.angulos_df.iloc[i]
            info = "DADOS ARTICULARES\n" + "="*17 + "\n\n"
            for l, l_full in [('D', 'DIREITO (Vermelho)'), ('E', 'ESQUERDO (Azul)')]:
                info += f">>> LADO {l_full}\n\n"
                for j, j_full in [('Quad', 'Quadril'), ('Joel', 'Joelho'), ('Torn', 'Tornozelo')]:
                    s = stats_ang.get(f'{j}_{l}', {'min':0, 'max':0})
                    info += f"{j_full}:\n  Atual: {row[f'{j}_{l}']:>5.1f}°\n  Mín: {s['min']:>4.0f}° | Máx: {s['max']:>4.0f}°\n\n"
            t_dynamic.set_text(info)

            if i < self.proc.n_frames - 1:
                p_prox = self.proc.segmentos_df.iloc[i+1]; p_curr = self.proc.segmentos_df.iloc[i]
                pares = [('Coxa_D', 'Perna_D', ptr_qjd, txt_qj_d), ('Perna_D', 'Pe_D', ptr_jtd, txt_jt_d), ('Coxa_E', 'Perna_E', ptr_qje, txt_qj_e), ('Perna_E', 'Pe_E', ptr_jte, txt_jt_e)]
                for j_prox, j_dist, ptr, txt in pares:
                    dx, dy = p_prox[j_prox] - p_curr[j_prox], p_prox[j_dist] - p_curr[j_dist]
                    ang = np.degrees(np.arctan2(dy, dx)) % 360 if not (np.isnan(dx) or np.isnan(dy)) else np.nan
                    if not np.isnan(ang):
                        ptr.set_data([0, np.cos(np.radians(ang))], [0, np.sin(np.radians(ang))])
                        label, cor = self._classificar_angulo(ang)
                        txt.set_text(label); txt.set_color(cor)

            return list(linhas.values()) + [t_dynamic, txt_qj_d, txt_jt_d, txt_qj_e, txt_jt_e, ptr_qjd, ptr_jtd, ptr_qje, ptr_jte]

        ani = animation.FuncAnimation(fig, update, frames=range(0, self.proc.n_frames, step), interval=50)
        try:
            ani.save(caminho_final, writer='pillow', fps=fps_anim)
            return True, caminho_final
        except Exception as e: return False, str(e)
        finally: plt.close(fig); plt.close('all')

# =============================================================================
# INTERFACE WEB STREAMLIT
# =============================================================================
st.title("🚶 GPBIO - Sistema de Análise de Marcha")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📏 Dados Antropométricos")
st.sidebar.info("Planilha com colunas 'ID' e 'Altura' (em metros).")
arquivo_antropo = st.sidebar.file_uploader("Planilha de Altura (Excel/CSV)", type=['xlsx', 'csv'])

df_antropo = None
if arquivo_antropo:
    if arquivo_antropo.name.endswith('xlsx'): df_antropo = pd.read_excel(arquivo_antropo)
    else: df_antropo = pd.read_csv(arquivo_antropo, sep=';', decimal=',')
    df_antropo.columns = df_antropo.columns.str.strip().str.upper()
    if 'ID' in df_antropo.columns and 'ALTURA' in df_antropo.columns:
        df_antropo['ID'] = df_antropo['ID'].astype(str).str.upper().str.strip()
        st.sidebar.success(f"Dados de {len(df_antropo)} participantes prontos!")
    else:
        st.sidebar.error("Erro nas colunas.")
        df_antropo = None

st.sidebar.markdown("---")
st.sidebar.markdown("### Sobre o Sistema")
st.sidebar.info("GPBIO: Análise Biomecânica de Marcha.")
st.sidebar.markdown("**Desenvolvido por Arthur Lins**")
	
if 'processadores' not in st.session_state: st.session_state.processadores = []

st.subheader("📁 Importação de Dados")
col_g1, col_g2 = st.columns(2)
with col_g1:
    nome_g1 = st.text_input("Nome do Grupo 1", value="Controle")
    files_g1 = st.file_uploader(f"C3D - {nome_g1}", type=['c3d'], accept_multiple_files=True, key="up_g1")
with col_g2:
    nome_g2 = st.text_input("Nome do Grupo 2", value="Parkinson")
    files_g2 = st.file_uploader(f"C3D - {nome_g2}", type=['c3d'], accept_multiple_files=True, key="up_g2")

if st.button("Processar Arquivos", type="primary", use_container_width=True):
    arquivos_para_processar = []
    if files_g1: arquivos_para_processar.extend([(f, nome_g1) for f in files_g1 if "CAL" not in f.name.upper()])
    if files_g2: arquivos_para_processar.extend([(f, nome_g2) for f in files_g2 if "CAL" not in f.name.upper()])

    if not arquivos_para_processar:
        st.warning("Faça o upload.")
    else:
        st.session_state.processadores = []
        progress_bar = st.progress(0)
        
        for i, (file, nome_grupo) in enumerate(arquivos_para_processar):
            file.seek(0) 
            with tempfile.NamedTemporaryFile(delete=False, suffix='.c3d') as tmp_file:
                tmp_file.write(file.read()); tmp_file.flush(); os.fsync(tmp_file.fileno()); tmp_path = tmp_file.name
                
            proc = ProcessadorCinematico(tmp_path, file.name, grupo=nome_grupo, df_antropo=df_antropo)
            if proc.valido: st.session_state.processadores.append(proc)
            else: st.error(f"Erro: {proc.erro_msg}")
            try: os.remove(tmp_path)
            except Exception: pass 
            progress_bar.progress((i + 1) / len(arquivos_para_processar))
            
        st.success(f"✅ {len(st.session_state.processadores)} processados!")

if st.session_state.processadores:
    # Preparação Global
    grupos_estudo = sorted(list(set([p.grupo for p in st.session_state.processadores])))
    cores_comp = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#e377c2']
    
    def obter_estilo(grp, idx):
        g = grp.lower()
        if 'control' in g: return 'black', '-', 1.2
        if 'parkinson' in g: return 'black', '--', 1.2
        return cores_comp[idx % len(cores_comp)], '--', 1.2

    def calcular_ca_serie(prox_cics, dist_cics):
        cas = []
        for cp, cd in zip(prox_cics, dist_cics):
            # Fórmula Vector Coding baseada na literatura
            ca = np.mod(np.degrees(np.arctan2(np.diff(cd), np.diff(cp))), 360)
            ca = np.append(ca, ca[-1]) 
            cas.append(ca)
        return cas

    def media_circular(ciclos):
        if not ciclos: return []
        rad = np.radians(np.array(ciclos))
        s_m = np.nanmean(np.sin(rad), axis=0)
        c_m = np.nanmean(np.cos(rad), axis=0)
        return np.mod(np.degrees(np.arctan2(s_m, c_m)), 360)

    dados_curvas = {grp: {'Art': {}, 'Seg': {}, 'CA_Art': {}, 'CA_Seg': {}} for grp in grupos_estudo}
    chaves_art = ['Quad_D', 'Joel_D', 'Torn_D', 'Quad_E', 'Joel_E', 'Torn_E']
    chaves_seg = ['Coxa_D', 'Perna_D', 'Pe_D', 'Coxa_E', 'Perna_E', 'Pe_E']
    chaves_ca_art = ['Quad_Joel_D', 'Quad_Joel_E', 'Joel_Torn_D', 'Joel_Torn_E']
    chaves_ca_seg = ['Coxa_Perna_D', 'Coxa_Perna_E', 'Perna_Pe_D', 'Perna_Pe_E']
    
    for grp in grupos_estudo:
        for c in chaves_art: dados_curvas[grp]['Art'][c] = []
        for c in chaves_seg: dados_curvas[grp]['Seg'][c] = []
        for c in chaves_ca_art: dados_curvas[grp]['CA_Art'][c] = []
        for c in chaves_ca_seg: dados_curvas[grp]['CA_Seg'][c] = []
        
        procs_grp = [p for p in st.session_state.processadores if p.grupo == grp]
        for p in procs_grp:
            for lado in ['D', 'E']:
                hss = p.eventos[lado]['HS']
                cics_quad = p.extrair_ciclos_normalizados(p.angulos_df[f"Quad_{lado}"].values, hss)
                cics_joel = p.extrair_ciclos_normalizados(p.angulos_df[f"Joel_{lado}"].values, hss)
                cics_torn = p.extrair_ciclos_normalizados(p.angulos_df[f"Torn_{lado}"].values, hss)
                cics_coxa = p.extrair_ciclos_normalizados(p.segmentos_df[f"Coxa_{lado}"].values, hss)
                cics_perna = p.extrair_ciclos_normalizados(p.segmentos_df[f"Perna_{lado}"].values, hss)
                cics_pe = p.extrair_ciclos_normalizados(p.segmentos_df[f"Pe_{lado}"].values, hss)
                
                dados_curvas[grp]['Art'][f"Quad_{lado}"].extend(cics_quad)
                dados_curvas[grp]['Art'][f"Joel_{lado}"].extend(cics_joel)
                dados_curvas[grp]['Art'][f"Torn_{lado}"].extend(cics_torn)
                dados_curvas[grp]['Seg'][f"Coxa_{lado}"].extend(cics_coxa)
                dados_curvas[grp]['Seg'][f"Perna_{lado}"].extend(cics_perna)
                dados_curvas[grp]['Seg'][f"Pe_{lado}"].extend(cics_pe)
                
                dados_curvas[grp]['CA_Art'][f"Quad_Joel_{lado}"].extend(calcular_ca_serie(cics_quad, cics_joel))
                dados_curvas[grp]['CA_Art'][f"Joel_Torn_{lado}"].extend(calcular_ca_serie(cics_joel, cics_torn))
                dados_curvas[grp]['CA_Seg'][f"Coxa_Perna_{lado}"].extend(calcular_ca_serie(cics_coxa, cics_perna))
                dados_curvas[grp]['CA_Seg'][f"Perna_Pe_{lado}"].extend(calcular_ca_serie(cics_perna, cics_pe))

    tab1, tab2, tab_ca, tab3, tab4, tab5 = st.tabs([
        "📊 Tabela de Médias", "📈 Cinemática Art/Seg", "📈 Coupling Angle", "⚙️ Coordenação Vetorial", 
        "🎥 Animações 3D", "📦 Estatística"
    ])

    with tab1:
        st.subheader("📊 Tabela de Dados Agrupados (Média por Paciente)")
        dados_tabela = []
        for p in st.session_state.processadores:
            try:
                linha = {
                    "ID_Paciente": p.id_paciente, "Grupo": p.grupo, "Velocidade (m/s)": getattr(p, 'velocidade_media', np.nan),
                    "Apoio DIR (%)": p.fases_marcha.get('D', {}).get('Apoio', np.nan), "Apoio ESQ (%)": p.fases_marcha.get('E', {}).get('Apoio', np.nan),
                    "Clearance DIR (mm)": p.foot_clearance.get('D', np.nan), "Clearance ESQ (mm)": p.foot_clearance.get('E', np.nan),
                    "Passo DIR (mm)": p.comprimento_passo.get('D', np.nan), "Passo ESQ (mm)": p.comprimento_passo.get('E', np.nan),
                    "Passo DIR (% Altura)": p.passo_norm.get('D', np.nan) if hasattr(p, 'passo_norm') else np.nan,
                    "Passo ESQ (% Altura)": p.passo_norm.get('E', np.nan) if hasattr(p, 'passo_norm') else np.nan,
                }
                
                stats = p.obter_stats()
                if stats:
                    for art in ['Quad_D', 'Quad_E', 'Joel_D', 'Joel_E', 'Torn_D', 'Torn_E']:
                        linha[f"{art} Máx (°)"] = stats.get(art, {}).get('max', np.nan)
                        linha[f"{art} Mín (°)"] = stats.get(art, {}).get('min', np.nan)

                pares_coord_tab1 = [
                    ('Artic_QJ_DIR', 'Quad_Joel_D', 'angulos_df'), ('Artic_QJ_ESQ', 'Quad_Joel_E', 'angulos_df'),
                    ('Artic_JT_DIR', 'Joel_Torn_D', 'angulos_df'), ('Artic_JT_ESQ', 'Joel_Torn_E', 'angulos_df'),
                    ('Segm_CP_DIR', 'Coxa_Perna_D', 'segmentos_df'), ('Segm_CP_ESQ', 'Coxa_Perna_E', 'segmentos_df'),
                    ('Segm_PP_DIR', 'Perna_Pe_D', 'segmentos_df'), ('Segm_PP_ESQ', 'Perna_Pe_E', 'segmentos_df')
                ]
                padroes = ['Proximal', 'EmFase', 'Distal', 'AntiFase']

                for par_label, par_key, nome_df in pares_coord_tab1:
                    try:
                        prox_name, dist_name, lado = par_key.split('_')[0], par_key.split('_')[1], par_key.split('_')[2]
                        hss = p.eventos[lado]['HS']
                        df_obj = getattr(p, nome_df)
                        
                        if len(hss) > 1:
                            c_prox = p.extrair_ciclos_normalizados(df_obj[f"{prox_name}_{lado}"].values, hss)
                            c_dist = p.extrair_ciclos_normalizados(df_obj[f"{dist_name}_{lado}"].values, hss)
                            
                            if len(c_prox) > 0 and len(c_dist) > 0:
                                arr_p, arr_d = np.array(c_prox), np.array(c_dist)
                                delta_p, delta_d = np.diff(arr_p, axis=1), np.diff(arr_d, axis=1)
                                gamma_rad = np.arctan2(delta_d, delta_p)
                                gamma_deg = (np.degrees(gamma_rad) + 360) % 360
                                
                                x_m, y_m = np.mean(np.cos(gamma_rad), axis=0), np.mean(np.sin(gamma_rad), axis=0)
                                r = np.clip(np.sqrt(x_m**2 + y_m**2), 0, 1)
                                linha[f"CAV {par_label} (°)"] = np.mean(np.sqrt(2 * (1 - r)) * (180 / np.pi))
                                
                                padroes_idx = np.digitize(gamma_deg, [0, 45, 135, 225, 315, 360])
                                padroes_idx[padroes_idx == 5] = 1
                                linha[f"Transições {par_label}"] = np.mean(np.sum(np.diff(padroes_idx, axis=1) != 0, axis=1))
                            else:
                                linha[f"CAV {par_label} (°)"] = np.nan; linha[f"Transições {par_label}"] = np.nan
                        else:
                            linha[f"CAV {par_label} (°)"] = np.nan; linha[f"Transições {par_label}"] = np.nan

                        serie = p.coord_vetorial_series.get(par_key, [])
                        fatia_apoio = serie[0:60] if len(serie) >= 60 else []
                        fatia_balanco = serie[60:] if len(serie) > 60 else []

                        for padrao in padroes:
                            linha[f"APOIO {par_label} - {padrao} (%)"] = (fatia_apoio.count(padrao) / len(fatia_apoio)) * 100 if len(fatia_apoio) > 0 else np.nan
                            linha[f"BALANÇO {par_label} - {padrao} (%)"] = (fatia_balanco.count(padrao) / len(fatia_balanco)) * 100 if len(fatia_balanco) > 0 else np.nan
                    except Exception:
                        linha[f"CAV {par_label} (°)"] = np.nan; linha[f"Transições {par_label}"] = np.nan
                dados_tabela.append(linha)
            except Exception: continue
                
        if dados_tabela:
            df_bruto = pd.DataFrame(dados_tabela)
            cols_num = df_bruto.select_dtypes(include=[np.number]).columns.tolist()
            df_agrupado_pacientes = df_bruto.groupby(['Grupo', 'ID_Paciente'])[cols_num].mean().reset_index().round(2).replace(np.nan, "")
            st.dataframe(df_agrupado_pacientes, use_container_width=True, height=600)
            csv = df_agrupado_pacientes.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
            st.download_button("📥 Baixar Tabela Agrupada", data=csv, file_name="estatistica_agrupada.csv", mime="text/csv", type="primary")

    with tab2:
        st.subheader("📈 Curvas Cinemáticas Normalizadas (0-100% do Ciclo)")
        x_axis = np.linspace(0, 100, 101)

        sub_t1, sub_t2, sub_t3, sub_t4 = st.tabs(["🟢 Normativo (Controle)", "⚖️ Comparação Articular", "⚖️ Comparação Segmentar", "🔍 Curvas Individuais"])

        with sub_t1:
            st.markdown("<h5 style='text-align:center;'>Média Bilateral Isolada - Grupo Controle</h5>", unsafe_allow_html=True)
            grupo_controle = [g for g in grupos_estudo if 'control' in g.lower()]
            if grupo_controle:
                grp_ctrl = grupo_controle[0]
                fig_ctrl, axs_ctrl = plt.subplots(2, 3, figsize=(15, 8))
                
                articulacoes = ['Quad', 'Joel', 'Torn']
                titulos_art = ['Quadril Articular', 'Joelho Articular', 'Tornozelo Articular']
                for i, art in enumerate(articulacoes):
                    ax = axs_ctrl[0, i]
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.set_title(titulos_art[i], fontweight='bold', fontsize=11)
                    if i == 0: ax.set_ylabel("Graus (°)", fontsize=9)
                    ax.axhline(0, color='black', lw=0.8)
                    ciclos = dados_curvas[grp_ctrl]['Art'][f"{art}_D"] + dados_curvas[grp_ctrl]['Art'][f"{art}_E"]
                    if ciclos: ax.plot(x_axis, np.mean(np.array(ciclos), axis=0), color='black', lw=1.2)

                segmentos = ['Coxa', 'Perna', 'Pe']
                titulos_seg = ['Coxa Segmentar', 'Perna Segmentar', 'Pé Segmentar']
                for i, seg in enumerate(segmentos):
                    ax = axs_ctrl[1, i]
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.set_title(titulos_seg[i], fontweight='bold', fontsize=11)
                    ax.set_xlabel("% Ciclo", fontsize=9)
                    if i == 0: ax.set_ylabel("Graus (°)", fontsize=9)
                    ax.axhline(0, color='black', lw=0.8)
                    ciclos = dados_curvas[grp_ctrl]['Seg'][f"{seg}_D"] + dados_curvas[grp_ctrl]['Seg'][f"{seg}_E"]
                    if ciclos: ax.plot(x_axis, np.mean(np.array(ciclos), axis=0), color='black', lw=1.2)

                plt.tight_layout(); st.pyplot(fig_ctrl); plt.close(fig_ctrl)

        with sub_t2:
            st.markdown("<h5 style='text-align:center;'>Comparativo Articular Bilateral</h5>", unsafe_allow_html=True)
            fig_comp_art, axs_comp_art = plt.subplots(1, 3, figsize=(15, 4.5))
            articulacoes = ['Quad', 'Joel', 'Torn']
            titulos = ['Quadril', 'Joelho', 'Tornozelo']

            for i, art in enumerate(articulacoes):
                ax = axs_comp_art[i]
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_title(titulos[i], fontweight='bold', fontsize=11)
                ax.set_xlabel("% Ciclo", fontsize=9)
                if i == 0: ax.set_ylabel("Graus (°)", fontsize=9)
                ax.axhline(0, color='black', lw=0.8)

                for idx, grp in enumerate(grupos_estudo):
                    ciclos_bilaterais = dados_curvas[grp]['Art'][f"{art}_D"] + dados_curvas[grp]['Art'][f"{art}_E"]
                    if ciclos_bilaterais:
                        cor, ls, lw = obter_estilo(grp, idx)
                        ax.plot(x_axis, np.mean(np.array(ciclos_bilaterais), axis=0), label=grp, color=cor, linestyle=ls, lw=lw)
                if i == 2: ax.legend(loc='best')
            plt.tight_layout(); st.pyplot(fig_comp_art); plt.close(fig_comp_art)

        with sub_t3:
            st.markdown("<h5 style='text-align:center;'>Comparativo Segmentar Bilateral</h5>", unsafe_allow_html=True)
            fig_comp_seg, axs_comp_seg = plt.subplots(1, 3, figsize=(15, 4.5))
            segmentos = ['Coxa', 'Perna', 'Pe']
            titulos_seg = ['Coxa', 'Perna', 'Pé']

            for i, seg in enumerate(segmentos):
                ax = axs_comp_seg[i]
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_title(titulos_seg[i], fontweight='bold', fontsize=11)
                ax.set_xlabel("% Ciclo", fontsize=9)
                if i == 0: ax.set_ylabel("Graus (°)", fontsize=9)
                ax.axhline(0, color='black', lw=0.8)

                for idx, grp in enumerate(grupos_estudo):
                    ciclos_bilaterais = dados_curvas[grp]['Seg'][f"{seg}_D"] + dados_curvas[grp]['Seg'][f"{seg}_E"]
                    if ciclos_bilaterais:
                        cor, ls, lw = obter_estilo(grp, idx)
                        ax.plot(x_axis, np.mean(np.array(ciclos_bilaterais), axis=0), label=grp, color=cor, linestyle=ls, lw=lw)
                if i == 2: ax.legend(loc='best')
            plt.tight_layout(); st.pyplot(fig_comp_seg); plt.close(fig_comp_seg)

        with sub_t4:
            cols_sep = st.columns(len(grupos_estudo))
            for idx, grp in enumerate(grupos_estudo):
                with cols_sep[idx]:
                    st.markdown(f"<h5 style='text-align:center;'>Grupo: {grp}</h5>", unsafe_allow_html=True)
                    fig_ind, axs_ind = plt.subplots(6, 2, figsize=(7, 18), sharex=True)
                    mapeamento_ind = [
                        ('Quad_D', 0, 0, 'Quad Artic (DIR)', 'Art'), ('Quad_E', 0, 1, 'Quad Artic (ESQ)', 'Art'),
                        ('Joel_D', 1, 0, 'Joel Artic (DIR)', 'Art'), ('Joel_E', 1, 1, 'Joel Artic (ESQ)', 'Art'),
                        ('Torn_D', 2, 0, 'Torn Artic (DIR)', 'Art'), ('Torn_E', 2, 1, 'Torn Artic (ESQ)', 'Art'),
                        ('Coxa_D', 3, 0, 'Coxa Segm (DIR)', 'Seg'), ('Coxa_E', 3, 1, 'Coxa Segm (ESQ)', 'Seg'),
                        ('Perna_D', 4, 0, 'Perna Segm (DIR)', 'Seg'), ('Perna_E', 4, 1, 'Perna Segm (ESQ)', 'Seg'),
                        ('Pe_D', 5, 0, 'Pé Segm (DIR)', 'Seg'), ('Pe_E', 5, 1, 'Pé Segm (ESQ)', 'Seg')
                    ]
                    cor, ls, lw = obter_estilo(grp, idx)
                    for chave, row, col, titulo, tipo in mapeamento_ind:
                        ax = axs_ind[row, col]
                        ciclos = np.array(dados_curvas[grp][tipo][chave])
                        ax.grid(True, linestyle='--', alpha=0.5)
                        ax.set_title(titulo, fontweight='bold', fontsize=10)
                        if col == 0: ax.set_ylabel("Graus (°)", fontsize=9)
                        if row == 5: ax.set_xlabel("% Ciclo", fontsize=9)

                        if len(ciclos) > 0:
                            ax.plot(x_axis, np.mean(ciclos, axis=0), color=cor, linestyle=ls, lw=lw)
                            ax.axhline(0, color='black', lw=0.8)
                    plt.tight_layout(); st.pyplot(fig_ind); plt.close(fig_ind)

    with tab_ca:
        st.subheader("📈 Coupling Angle - Séries Temporais (Vector Coding)")
        x_axis = np.linspace(0, 100, 101)

        sub_ca1, sub_ca2, sub_ca3, sub_ca4 = st.tabs(["🟢 Normativo (Controle)", "⚖️ Comparação Articular", "⚖️ Comparação Segmentar", "🔍 Curvas Individuais"])

        with sub_ca1:
            st.markdown("<h5 style='text-align:center;'>Média Bilateral Isolada - Grupo Controle</h5>", unsafe_allow_html=True)
            grupo_controle = [g for g in grupos_estudo if 'control' in g.lower()]
            if grupo_controle:
                grp_ctrl = grupo_controle[0]
                fig_ca_ctrl, axs_ca_ctrl = plt.subplots(2, 2, figsize=(10, 8))
                
                pares_ca_ctrl = [
                    (axs_ca_ctrl[0,0], 'Quad_Joel', 'CA_Art', 'Quadril-Joelho (°)'), (axs_ca_ctrl[0,1], 'Joel_Torn', 'CA_Art', 'Joelho-Tornozelo (°)'),
                    (axs_ca_ctrl[1,0], 'Coxa_Perna', 'CA_Seg', 'Coxa-Perna (°)'), (axs_ca_ctrl[1,1], 'Perna_Pe', 'CA_Seg', 'Perna-Pé (°)')
                ]
                
                for ax, par, tipo, titulo in pares_ca_ctrl:
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.set_title(titulo, fontweight='bold', fontsize=11)
                    ax.set_xlabel("% Ciclo", fontsize=9); ax.set_ylabel("Coupling Angle (°)", fontsize=9)
                    ax.set_ylim(0, 360); ax.set_yticks([0, 90, 180, 270, 360])
                    
                    ciclos = dados_curvas[grp_ctrl][tipo][f"{par}_D"] + dados_curvas[grp_ctrl][tipo][f"{par}_E"]
                    if ciclos: ax.plot(x_axis, media_circular(ciclos), color='black', lw=1.2)

                plt.tight_layout(); st.pyplot(fig_ca_ctrl); plt.close(fig_ca_ctrl)

        with sub_ca2:
            st.markdown("<h5 style='text-align:center;'>Comparativo Articular Bilateral (CA)</h5>", unsafe_allow_html=True)
            fig_ca_art, axs_ca_art = plt.subplots(1, 2, figsize=(10, 4.5))
            pares_ca_art = [(axs_ca_art[0], 'Quad_Joel', 'Quadril-Joelho (°)'), (axs_ca_art[1], 'Joel_Torn', 'Joelho-Tornozelo (°)')]

            for ax, par, titulo in pares_ca_art:
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_title(titulo, fontweight='bold', fontsize=11)
                ax.set_xlabel("% Ciclo", fontsize=9); ax.set_ylabel("Coupling Angle (°)", fontsize=9)
                ax.set_ylim(0, 360); ax.set_yticks([0, 90, 180, 270, 360])

                for idx, grp in enumerate(grupos_estudo):
                    ciclos = dados_curvas[grp]['CA_Art'][f"{par}_D"] + dados_curvas[grp]['CA_Art'][f"{par}_E"]
                    if ciclos:
                        cor, ls, lw = obter_estilo(grp, idx)
                        ax.plot(x_axis, media_circular(ciclos), label=grp, color=cor, linestyle=ls, lw=lw)
                ax.legend(loc='best')
            plt.tight_layout(); st.pyplot(fig_ca_art); plt.close(fig_ca_art)

        with sub_ca3:
            st.markdown("<h5 style='text-align:center;'>Comparativo Segmentar Bilateral (CA)</h5>", unsafe_allow_html=True)
            fig_ca_seg, axs_ca_seg = plt.subplots(1, 2, figsize=(10, 4.5))
            pares_ca_seg = [(axs_ca_seg[0], 'Coxa_Perna', 'Coxa-Perna (°)'), (axs_ca_seg[1], 'Perna_Pe', 'Perna-Pé (°)')]

            for ax, par, titulo in pares_ca_seg:
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_title(titulo, fontweight='bold', fontsize=11)
                ax.set_xlabel("% Ciclo", fontsize=9); ax.set_ylabel("Coupling Angle (°)", fontsize=9)
                ax.set_ylim(0, 360); ax.set_yticks([0, 90, 180, 270, 360])

                for idx, grp in enumerate(grupos_estudo):
                    ciclos = dados_curvas[grp]['CA_Seg'][f"{par}_D"] + dados_curvas[grp]['CA_Seg'][f"{par}_E"]
                    if ciclos:
                        cor, ls, lw = obter_estilo(grp, idx)
                        ax.plot(x_axis, media_circular(ciclos), label=grp, color=cor, linestyle=ls, lw=lw)
                ax.legend(loc='best')
            plt.tight_layout(); st.pyplot(fig_ca_seg); plt.close(fig_ca_seg)

        with sub_ca4:
            cols_sep_ca = st.columns(len(grupos_estudo))
            for idx, grp in enumerate(grupos_estudo):
                with cols_sep_ca[idx]:
                    st.markdown(f"<h5 style='text-align:center;'>Grupo: {grp}</h5>", unsafe_allow_html=True)
                    fig_ind_ca, axs_ind_ca = plt.subplots(4, 2, figsize=(7, 12), sharex=True)
                    mapeamento_ca_ind = [
                        ('Quad_Joel_D', 0, 0, 'Q-J (DIR)', 'CA_Art'), ('Quad_Joel_E', 0, 1, 'Q-J (ESQ)', 'CA_Art'),
                        ('Joel_Torn_D', 1, 0, 'J-T (DIR)', 'CA_Art'), ('Joel_Torn_E', 1, 1, 'J-T (ESQ)', 'CA_Art'),
                        ('Coxa_Perna_D', 2, 0, 'C-P (DIR)', 'CA_Seg'), ('Coxa_Perna_E', 2, 1, 'C-P (ESQ)', 'CA_Seg'),
                        ('Perna_Pe_D', 3, 0, 'P-P (DIR)', 'CA_Seg'), ('Perna_Pe_E', 3, 1, 'P-P (ESQ)', 'CA_Seg')
                    ]
                    
                    cor, ls, lw = obter_estilo(grp, idx)

                    for chave, row, col, titulo, tipo in mapeamento_ca_ind:
                        ax = axs_ind_ca[row, col]
                        ciclos = dados_curvas[grp][tipo][chave]
                        ax.grid(True, linestyle='--', alpha=0.5)
                        ax.set_title(titulo, fontweight='bold', fontsize=10)
                        if col == 0: ax.set_ylabel("CA (°)", fontsize=9)
                        if row == 3: ax.set_xlabel("% Ciclo", fontsize=9)
                        ax.set_ylim(0, 360); ax.set_yticks([0, 180, 360])

                        if len(ciclos) > 0:
                            ax.plot(x_axis, media_circular(ciclos), color=cor, linestyle=ls, lw=lw)
                    plt.tight_layout(); st.pyplot(fig_ind_ca); plt.close(fig_ind_ca)

    with tab3:
        st.subheader("⚙️ Coordenação Vetorial e Controle Motor")
        st.markdown("### 1. Comportamento Espacial (Diagramas Angle-Angle)")
        
        sub_aa1, sub_aa2, sub_aa3, sub_aa4 = st.tabs(["🟢 Normativo (Controle)", "⚖️ Comparação Articular", "⚖️ Comparação Segmentar", "🔍 Curvas Individuais"])

        with sub_aa1:
            st.markdown("<h5 style='text-align:center;'>Média Bilateral Isolada - Grupo Controle</h5>", unsafe_allow_html=True)
            grupo_controle = [g for g in grupos_estudo if 'control' in g.lower()]
            if grupo_controle:
                grp = grupo_controle[0]
                fig_aa_ctrl, axs_aa_ctrl = plt.subplots(2, 2, figsize=(9, 9))
                
                pares_norm_aa = [
                    (axs_aa_ctrl[0,0], 'Quad', 'Joel', 'Art', 'Quadril (°)', 'Joelho (°)'),
                    (axs_aa_ctrl[0,1], 'Joel', 'Torn', 'Art', 'Joelho (°)', 'Tornozelo (°)'),
                    (axs_aa_ctrl[1,0], 'Coxa', 'Perna', 'Seg', 'Coxa (°)', 'Perna (°)'),
                    (axs_aa_ctrl[1,1], 'Perna', 'Pe', 'Seg', 'Perna (°)', 'Pé (°)')
                ]
                
                for ax, x_k, y_k, tipo, label_x, label_y in pares_norm_aa:
                    x_cics = dados_curvas[grp][tipo][f"{x_k}_D"] + dados_curvas[grp][tipo][f"{x_k}_E"]
                    y_cics = dados_curvas[grp][tipo][f"{y_k}_D"] + dados_curvas[grp][tipo][f"{y_k}_E"]
                    if x_cics and y_cics:
                        x_mean, y_mean = np.mean(np.array(x_cics), axis=0), np.mean(np.array(y_cics), axis=0)
                        ax.plot(x_mean, y_mean, color='black', lw=1.2)
                        ax.scatter(x_mean[0], y_mean[0], color='green', s=40, zorder=5)
                        ax.scatter(x_mean[60], y_mean[60], color='orange', marker='X', s=40, zorder=5)
                    ax.set_xlabel(label_x, fontsize=9); ax.set_ylabel(label_y, fontsize=9)
                    ax.set_title(f"{x_k}-{y_k}", fontweight='bold', fontsize=11)
                    ax.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout(); st.pyplot(fig_aa_ctrl); plt.close(fig_aa_ctrl)

        with sub_aa2:
            st.markdown("<h5 style='text-align:center;'>Comparativo Articular Bilateral</h5>", unsafe_allow_html=True)
            fig_aa_art, axs_aa_art = plt.subplots(1, 2, figsize=(10, 5))
            pares_comp_art = [(axs_aa_art[0], 'Quad', 'Joel', 'Quadril (°)', 'Joelho (°)'), (axs_aa_art[1], 'Joel', 'Torn', 'Joelho (°)', 'Tornozelo (°)')]

            for ax, x_k, y_k, label_x, label_y in pares_comp_art:
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_title(f"{x_k}-{y_k}", fontweight='bold', fontsize=11)
                ax.set_xlabel(label_x, fontsize=9); ax.set_ylabel(label_y, fontsize=9)
                
                for idx, grp in enumerate(grupos_estudo):
                    x_cics = dados_curvas[grp]['Art'][f"{x_k}_D"] + dados_curvas[grp]['Art'][f"{x_k}_E"]
                    y_cics = dados_curvas[grp]['Art'][f"{y_k}_D"] + dados_curvas[grp]['Art'][f"{y_k}_E"]
                    if x_cics and y_cics:
                        cor, ls, lw = obter_estilo(grp, idx)
                        x_mean, y_mean = np.mean(np.array(x_cics), axis=0), np.mean(np.array(y_cics), axis=0)
                        ax.plot(x_mean, y_mean, label=grp, color=cor, linestyle=ls, lw=lw)
                        ax.scatter(x_mean[0], y_mean[0], color='green', s=30, zorder=5)
                        ax.scatter(x_mean[60], y_mean[60], color='orange', marker='X', s=30, zorder=5)
                ax.legend(loc='best')
            plt.tight_layout(); st.pyplot(fig_aa_art); plt.close(fig_aa_art)

        with sub_aa3:
            st.markdown("<h5 style='text-align:center;'>Comparativo Segmentar Bilateral</h5>", unsafe_allow_html=True)
            fig_aa_seg, axs_aa_seg = plt.subplots(1, 2, figsize=(10, 5))
            pares_comp_seg = [(axs_aa_seg[0], 'Coxa', 'Perna', 'Coxa (°)', 'Perna (°)'), (axs_aa_seg[1], 'Perna', 'Pe', 'Perna (°)', 'Pé (°)')]

            for ax, x_k, y_k, label_x, label_y in pares_comp_seg:
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_title(f"{x_k}-{y_k}", fontweight='bold', fontsize=11)
                ax.set_xlabel(label_x, fontsize=9); ax.set_ylabel(label_y, fontsize=9)
                
                for idx, grp in enumerate(grupos_estudo):
                    x_cics = dados_curvas[grp]['Seg'][f"{x_k}_D"] + dados_curvas[grp]['Seg'][f"{x_k}_E"]
                    y_cics = dados_curvas[grp]['Seg'][f"{y_k}_D"] + dados_curvas[grp]['Seg'][f"{y_k}_E"]
                    if x_cics and y_cics:
                        cor, ls, lw = obter_estilo(grp, idx)
                        x_mean, y_mean = np.mean(np.array(x_cics), axis=0), np.mean(np.array(y_cics), axis=0)
                        ax.plot(x_mean, y_mean, label=grp, color=cor, linestyle=ls, lw=lw)
                        ax.scatter(x_mean[0], y_mean[0], color='green', s=30, zorder=5)
                        ax.scatter(x_mean[60], y_mean[60], color='orange', marker='X', s=30, zorder=5)
                ax.legend(loc='best')
            plt.tight_layout(); st.pyplot(fig_aa_seg); plt.close(fig_aa_seg)

        with sub_aa4:
            cols_ind_aa = st.columns(len(grupos_estudo))
            for idx, grp in enumerate(grupos_estudo):
                with cols_ind_aa[idx]:
                    st.markdown(f"<h5 style='text-align:center;'>Grupo: {grp}</h5>", unsafe_allow_html=True)
                    fig_ind_aa, axs_ind_aa = plt.subplots(4, 2, figsize=(7, 14))
                    mapeamento_aa_ind = [
                        (axs_ind_aa[0,0], 'Quad_D', 'Joel_D', 'Art', 'Quad(°)', 'Joel(°) (DIR)'), 
                        (axs_ind_aa[0,1], 'Quad_E', 'Joel_E', 'Art', 'Quad(°)', 'Joel(°) (ESQ)'),
                        (axs_ind_aa[1,0], 'Joel_D', 'Torn_D', 'Art', 'Joel(°)', 'Torn(°) (DIR)'), 
                        (axs_ind_aa[1,1], 'Joel_E', 'Torn_E', 'Art', 'Joel(°)', 'Torn(°) (ESQ)'),
                        (axs_ind_aa[2,0], 'Coxa_D', 'Perna_D', 'Seg', 'Coxa(°)', 'Perna(°) (DIR)'), 
                        (axs_ind_aa[2,1], 'Coxa_E', 'Perna_E', 'Seg', 'Coxa(°)', 'Perna(°) (ESQ)'),
                        (axs_ind_aa[3,0], 'Perna_D', 'Pe_D', 'Seg', 'Perna(°)', 'Pé(°) (DIR)'), 
                        (axs_ind_aa[3,1], 'Perna_E', 'Pe_E', 'Seg', 'Perna(°)', 'Pé(°) (ESQ)')
                    ]
                    cor, ls, lw = obter_estilo(grp, idx)

                    for ax, x_k, y_k, tipo, lx, ly in mapeamento_aa_ind:
                        x_cics, y_cics = dados_curvas[grp][tipo][x_k], dados_curvas[grp][tipo][y_k]
                        if x_cics and y_cics:
                            xm, ym = np.mean(x_cics, axis=0), np.mean(y_cics, axis=0)
                            ax.plot(xm, ym, color=cor, linestyle=ls, lw=lw)
                            ax.scatter(xm[0], ym[0], color='green', s=30, zorder=5)
                            ax.scatter(xm[60], ym[60], color='orange', marker='X', s=30, zorder=5)
                        ax.set_xlabel(lx, fontsize=8); ax.set_ylabel(ly, fontsize=8)
                        ax.grid(True, linestyle='--', alpha=0.5)
                    plt.tight_layout(); st.pyplot(fig_ind_aa); plt.close(fig_ind_aa)

        st.markdown("---")
        st.markdown("### 2. Análise Quantitativa de Fases (Frequência de Ocorrência)")
        tipo_coord = st.radio("Selecione o Modelo Biomecânico para as Barras:", 
                              ["📐 Coordenação Segmentar", "📍 Coordenação Articular"], horizontal=True)
        
        fase_selecionada = st.radio("Selecione a Fase da Marcha:", ["Apoio (0-60%)", "Balanço (60-100%)"], horizontal=True)
        inicio_f, fim_f = (0, 60) if "Apoio" in fase_selecionada else (60, 100)

        # Definição visual em tons de cinza/preto-e-branco com hachuras (Hatches)
        padroes_bw = {
            'Proximal': {'color': 'white', 'hatch': '///'},
            'EmFase': {'color': 'lightgray', 'hatch': 'xxx'},
            'Distal': {'color': 'darkgray', 'hatch': '...'},
            'AntiFase': {'color': 'dimgray', 'hatch': '\\\\\\'}
        }

        if "Segmentar" in tipo_coord:
            pares_map = [('Coxa_Perna_D', 'Coxa-Perna (DIR)'), ('Coxa_Perna_E', 'Coxa-Perna (ESQ)'), ('Perna_Pe_D', 'Perna-Pé (DIR)'), ('Perna_Pe_E', 'Perna-Pé (ESQ)')]
        else:
            pares_map = [('Quad_Joel_D', 'Quad-Joel (DIR)'), ('Quad_Joel_E', 'Quad-Joel (ESQ)'), ('Joel_Torn_D', 'Joel-Torn (DIR)'), ('Joel_Torn_E', 'Joel-Torn (ESQ)')]

        freq_acumulada = {g: {c_new: {k: [] for k in padroes_bw} for _, c_new in pares_map} for g in grupos_estudo}

        for p in st.session_state.processadores:
            grp = p.grupo
            for c_old, c_new in pares_map:
                fatia = p.coord_vetorial_series.get(c_old, [])[inicio_f:fim_f]
                total_fatia = len(fatia)
                if total_fatia > 0:
                    freq_acumulada[grp][c_new]['Proximal'].append((fatia.count('Proximal') / total_fatia) * 100)
                    freq_acumulada[grp][c_new]['EmFase'].append((fatia.count('EmFase') / total_fatia) * 100)
                    freq_acumulada[grp][c_new]['Distal'].append((fatia.count('Distal') / total_fatia) * 100)
                    freq_acumulada[grp][c_new]['AntiFase'].append((fatia.count('AntiFase') / total_fatia) * 100)

        # Nova estrutura de Abas individuais solicitada para as barras
        sub_freq1, sub_freq2 = st.tabs(["🦵 Comparação Artic/Segm Lado Direito", "🦵 Comparação Artic/Segm Lado Esquerdo"])

        def plotar_barras_comparativas(container, chaves_filtro):
            cols_freq = container.columns(len(chaves_filtro))
            for idx, par_label in enumerate(chaves_filtro):
                with cols_freq[idx]:
                    fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
                    
                    x = np.arange(len(grupos_estudo))
                    width = 0.55
                    
                    m_prox = [np.mean(freq_acumulada[g][par_label]['Proximal']) if freq_acumulada[g][par_label]['Proximal'] else 0 for g in grupos_estudo]
                    m_fase = [np.mean(freq_acumulada[g][par_label]['EmFase']) if freq_acumulada[g][par_label]['EmFase'] else 0 for g in grupos_estudo]
                    m_dist = [np.mean(freq_acumulada[g][par_label]['Distal']) if freq_acumulada[g][par_label]['Distal'] else 0 for g in grupos_estudo]
                    m_anti = [np.mean(freq_acumulada[g][par_label]['AntiFase']) if freq_acumulada[g][par_label]['AntiFase'] else 0 for g in grupos_estudo]

                    b1 = ax_bar.bar(x, m_prox, width, label='Proximal', color=padroes_bw['Proximal']['color'], edgecolor='black', hatch=padroes_bw['Proximal']['hatch'])
                    b2 = ax_bar.bar(x, m_fase, width, bottom=m_prox, label='Em Fase', color=padroes_bw['EmFase']['color'], edgecolor='black', hatch=padroes_bw['EmFase']['hatch'])
                    b3 = ax_bar.bar(x, m_dist, width, bottom=np.add(m_prox, m_fase), label='Distal', color=padroes_bw['Distal']['color'], edgecolor='black', hatch=padroes_bw['Distal']['hatch'])
                    b4 = ax_bar.bar(x, m_anti, width, bottom=np.add(np.add(m_prox, m_fase), m_dist), label='Anti-Fase', color=padroes_bw['AntiFase']['color'], edgecolor='black', hatch=padroes_bw['AntiFase']['hatch'])

                    for bar_group in [b1, b2, b3, b4]:
                        lbls = [f"{v.get_height():.1f}%" if v.get_height() > 3.0 else "" for v in bar_group]
                        ax_bar.bar_label(bar_group, labels=lbls, label_type='center', color='black', fontweight='bold', fontsize=8, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
                    
                    ax_bar.set_title(par_label, fontweight='bold', fontsize=11)
                    ax_bar.set_ylabel('Frequência (%)', fontweight='bold')
                    ax_bar.set_xticks(x); ax_bar.set_xticklabels(grupos_estudo, fontweight='bold')
                    ax_bar.set_ylim(0, 105); ax_bar.spines['top'].set_visible(False); ax_bar.spines['right'].set_visible(False)
                    
                    if idx == len(chaves_filtro) - 1:
                        ax_bar.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title="Padrões").get_title().set_fontweight('bold')
                    
                    plt.tight_layout(); st.pyplot(fig_bar); plt.close(fig_bar)

        pares_direita = [p[1] for p in pares_map if "(DIR)" in p[1]]
        pares_esquerda = [p[1] for p in pares_map if "(ESQ)" in p[1]]

        plotar_barras_comparativas(sub_freq1, pares_direita)
        plotar_barras_comparativas(sub_freq2, pares_esquerda)

    with tab4:
        st.subheader("Gerador de GIFs e Biofeedback Visual")
        st.info("Escolha os arquivos que deseja animar e a velocidade de reprodução.")
        lista_nomes = [p.nome_arq for p in st.session_state.processadores]
        col_selecao, col_vel = st.columns([2, 1])
        with col_selecao: selecionados = st.multiselect("Selecione os arquivos para animar:", lista_nomes)
        with col_vel: vel_opcao = st.radio("Velocidade de Reprodução:", ["100% (Normal)", "75% (Lenta)", "50% (Muito Lenta)"])
        fps_escolhido = {"100% (Normal)": 20, "75% (Lenta)": 15, "50% (Muito Lenta)": 10}[vel_opcao]
        
        if st.button("Gerar GIFs Selecionados", type="primary"):
            for p in st.session_state.processadores:
                if p.nome_arq in selecionados:
                    with st.spinner(f"Processando animação ({vel_opcao}) para: {p.nome_arq}..."):
                        fd, tmp_gif_path = tempfile.mkstemp(suffix='.gif'); os.close(fd) 
                        viz = GeradorVisual(p, p.nome_arq)
                        sucesso, msg = viz.salvar(tmp_gif_path, step=3, fps_anim=fps_escolhido)
                        if sucesso:
                            st.image(tmp_gif_path, caption=f"Análise 3D: {p.nome_arq}", use_container_width=True)
                            with open(tmp_gif_path, "rb") as file_gif: st.download_button(f"📥 Baixar GIF", data=file_gif, file_name=f"{p.nome_arq.split('.')[0]}_3D.gif", mime="image/gif")
                        else: st.error(f"Falha: {msg}")

    with tab5:
        st.subheader("Estatística Espaço-Temporal")
        v_vel = {g: [] for g in grupos_estudo}; v_ap_d = {g: [] for g in grupos_estudo}; v_ap_e = {g: [] for g in grupos_estudo}
        v_fc_d = {g: [] for g in grupos_estudo}; v_fc_e = {g: [] for g in grupos_estudo}
        v_ps_d = {g: [] for g in grupos_estudo}; v_ps_e = {g: [] for g in grupos_estudo}
        v_ps_norm_d = {g: [] for g in grupos_estudo}; v_ps_norm_e = {g: [] for g in grupos_estudo}
        
        for p in st.session_state.processadores:
            grp = p.grupo
            if not np.isnan(p.velocidade_media): v_vel[grp].append(p.velocidade_media)
            if not np.isnan(p.fases_marcha['D']['Apoio']): v_ap_d[grp].append(p.fases_marcha['D']['Apoio'])
            if not np.isnan(p.fases_marcha['E']['Apoio']): v_ap_e[grp].append(p.fases_marcha['E']['Apoio'])
            if not np.isnan(p.foot_clearance['D']): v_fc_d[grp].append(p.foot_clearance['D'])
            if not np.isnan(p.foot_clearance['E']): v_fc_e[grp].append(p.foot_clearance['E'])
            if not np.isnan(p.comprimento_passo['D']): v_ps_d[grp].append(p.comprimento_passo['D'])
            if not np.isnan(p.comprimento_passo['E']): v_ps_e[grp].append(p.comprimento_passo['E'])
            if hasattr(p, 'passo_norm') and not np.isnan(p.passo_norm.get('D', np.nan)): v_ps_norm_d[grp].append(p.passo_norm['D'])
            if hasattr(p, 'passo_norm') and not np.isnan(p.passo_norm.get('E', np.nan)): v_ps_norm_e[grp].append(p.passo_norm['E'])

        def gerar_colunas_passo_norm(dict_dados, titulo):
            fig, ax = plt.subplots(figsize=(8, 6))
            labels = list(dict_dados.keys())
            means = [np.nanmean(dict_dados[l]) if dict_dados[l] else 0 for l in labels]
            stds = [np.nanstd(dict_dados[l]) if dict_dados[l] else 0 for l in labels]
            cores = ['#d3d3d3' if 'control' in l.lower() else ('#707070' if 'parkinson' in l.lower() else cores_comp[i % len(cores_comp)]) for i, l in enumerate(labels)]
            bars = ax.bar(labels, means, yerr=stds, capsize=10, color=cores, edgecolor='black', alpha=0.8)
            for i, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 1, f"Média: {means[i]:.0f}%\nDP: ±{stds[i]:.1f}%", ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
            ax.set_title(titulo, fontweight='bold', fontsize=14); ax.set_ylabel("Porcentagem da Estatura (%)", fontsize=12)
            ax.set_ylim(0, 100); ax.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout(); return fig
        
        def gerar_boxplot_isolado(dict_dados, titulo, ylabel):
            fig, ax = plt.subplots(figsize=(8, 6)) 
            labels = list(dict_dados.keys())
            dados_limpos = [dict_dados[l] for l in labels if len(dict_dados[l]) > 0]
            labels_limpos = [l for l in labels if len(dict_dados[l]) > 0]
            if dados_limpos:
                bp = ax.boxplot(dados_limpos, patch_artist=True, labels=labels_limpos)
                for i, patch in enumerate(bp['boxes']): 
                    grp_name = labels_limpos[i].split('(')[0] if '(' in labels_limpos[i] else labels_limpos[i]
                    patch.set_facecolor('#d3d3d3' if 'control' in grp_name.lower() else ('#707070' if 'parkinson' in grp_name.lower() else '#999999'))
                for median in bp['medians']: median.set(color='black', linewidth=2)
                for i, d in enumerate(dados_limpos):
                    media, dp, mediana = np.mean(d), np.std(d), np.median(d)
                    ax.text(i + 1.10, mediana, f"M: {media:.1f}\nDP: {dp:.1f}", ha='left', va='center', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'))
                ax.set_xlim(0.5, len(dados_limpos) + 0.9)
                ymin, ymax = ax.get_ylim(); ax.set_ylim(ymin - (ymax-ymin)*0.1, ymax + (ymax-ymin)*0.1)
                ax.set_title(titulo, fontweight='bold', fontsize=14); ax.set_ylabel(ylabel, fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout(); return fig

        dict_vel = {g: v_vel[g] for g in grupos_estudo}
        dict_ap = {}; dict_fc = {}; dict_ps = {}; dict_ps_norm = {}
        for g in grupos_estudo:
            dict_ap[f"{g}(D)"] = v_ap_d[g]; dict_ap[f"{g}(E)"] = v_ap_e[g]
            dict_fc[f"{g}(D)"] = v_fc_d[g]; dict_fc[f"{g}(E)"] = v_fc_e[g]
            dict_ps[f"{g}(D)"] = v_ps_d[g]; dict_ps[f"{g}(E)"] = v_ps_e[g]
            dict_ps_norm[f"{g}(D)"] = v_ps_norm_d[g]; dict_ps_norm[f"{g}(E)"] = v_ps_norm_e[g]
            
        col_box1, col_box2 = st.columns(2)
        with col_box1:
            fig1 = gerar_boxplot_isolado(dict_vel, "Velocidade Média", "m/s"); st.pyplot(fig1); plt.close(fig1)
            fig2 = gerar_boxplot_isolado(dict_fc, "Foot Clearance", "mm"); st.pyplot(fig2); plt.close(fig2)
        with col_box2:
            fig3 = gerar_boxplot_isolado(dict_ap, "Fase de Apoio", "% do Ciclo"); st.pyplot(fig3); plt.close(fig3)
            fig4 = gerar_boxplot_isolado(dict_ps, "Comprimento do Passo (Absoluto)", "mm"); st.pyplot(fig4); plt.close(fig4)
            
        st.markdown("---")
        col_box3, col_box4 = st.columns(2)
        with col_box3:
            fig5 = gerar_colunas_passo_norm(dict_ps_norm, "Comprimento do Passo (% Altura)"); st.pyplot(fig5); plt.close(fig5)
