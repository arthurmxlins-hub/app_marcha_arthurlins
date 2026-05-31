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
def ang_sagital(p_prox, p_dist, tipo='vertical'):
    """
    Calcula o ângulo absoluto do segmento no plano sagital preservando o sinal.
    - X cresce no eixo de progressão (para frente).
    - Z cresce para cima (vertical).
    """
    if p_prox is None or p_dist is None: 
        return np.nan
    
    if tipo == 'vertical':
        # 0° é a vertical (apontando para baixo). 
        # Valores positivos indicam que o segmento está projetado para a frente (flexão).
        # Valores negativos indicam extensão.
        dx = p_dist[0] - p_prox[0]
        dz = p_prox[2] - p_dist[2] 
        return np.degrees(np.arctan2(dx, dz))
    else:
        # 0° é a horizontal.
        # Valores positivos indicam inclinação do segmento para cima (Dorsiflexão).
        # Valores negativos indicam inclinação para baixo (Flexão Plantar).
        dx = p_dist[0] - p_prox[0]
        dz = p_dist[2] - p_prox[2]
        return np.degrees(np.arctan2(dz, dx))

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

            self.dados = self._filtrar_e_inverter()
            
            # --- MOTORES CINEMÁTICOS ---
            self.segmentos_df = self._calcular_angulos_segmentares() 
            self.angulos_df = self._calcular_angulos() 
            
            self.velocidade_media = self._calcular_velocidade_sacrum()
            self.eventos = self.detectar_eventos_zeni()
            self.fases_marcha = self._calcular_fases_marcha()
            self.foot_clearance = self._calcular_foot_clearance()
            self.comprimento_passo = self._calcular_comprimento_passo()
            self.tempo_ciclo_s = self._calcular_tempo_medio_ciclo()

            # --- NORMALIZAÇÃO ANTROPOMÉTRICA ---
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

            # --- MOTOR DE ASSIMETRIA ---
            self.indices_assimetria = {}
            pares = [
                ('Passo', self.passo_norm['D'], self.passo_norm['E']),
                ('Apoio', self.fases_marcha['D']['Apoio'], self.fases_marcha['E']['Apoio']),
                ('Clearance', self.foot_clearance['D'], self.foot_clearance['E'])
            ]
            for nome, d, e in pares:
                if not np.isnan(d) and not np.isnan(e) and (d + e) > 0:
                    self.indices_assimetria[nome] = (abs(d - e) / (0.5 * (d + e))) * 100.0
                else:
                    self.indices_assimetria[nome] = np.nan

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

    def _filtrar_e_inverter(self):
        d = self.dados_raw.copy()
        d[d==0.0] = np.nan
        nyq = 0.5 * self.freq
        b, a = signal.butter(4, 6.0/nyq, btype='low')
        out = np.zeros_like(d) * np.nan
        for m in range(d.shape[1]):
            for ax in range(3):
                sinal = d[ax, m, :]
                if np.isnan(sinal).all(): 
                    continue
                s_temp = pd.Series(sinal).interpolate(limit_direction='both').bfill().ffill()
                try:
                    filt = signal.filtfilt(b, a, s_temp.to_numpy())
                    out[ax, m, :] = filt
                except Exception: 
                    out[ax, m, :] = s_temp
        out[0, :, :] = -1 * out[0, :, :]
        return out

    def _calcular_angulos_segmentares(self):
        res = {k: [] for k in ['Coxa_D','Perna_D','Pe_D','Coxa_E','Perna_E','Pe_E']}
        for f in range(self.n_frames):
            for lado, l in [('D', 'R'), ('E', 'L')]:
                h = self._get(f'{l}IAS',f)
                k = self._mid(f'{l}LE', f'{l}ME',f)
                a = self._mid(f'{l}ML', f'{l}MM',f)
                p = self._mid(f'{l}FT1', f'{l}FT5',f)
                cal = self._get(f'{l}CAL',f)
                
                res[f'Coxa_{lado}'].append(ang_sagital(h, k, 'vertical'))
                res[f'Perna_{lado}'].append(ang_sagital(k, a, 'vertical'))
                res[f'Pe_{lado}'].append(ang_sagital(cal, p, 'horizontal'))
        return pd.DataFrame(res)

    def _calcular_angulos(self):
        res = {k: [] for k in ['Quad_D','Joel_D','Torn_D','Quad_E','Joel_E','Torn_E']}
        for f in range(self.n_frames):
            for lado in ['D', 'E']:
                coxa = self.segmentos_df[f'Coxa_{lado}'][f]
                perna = self.segmentos_df[f'Perna_{lado}'][f]
                pe = self.segmentos_df[f'Pe_{lado}'][f]
                
                # Respeitando a lógica Flexão (+) / Extensão (-)
                res[f'Quad_{lado}'].append(coxa) 
                res[f'Joel_{lado}'].append(coxa - perna if not (np.isnan(coxa) or np.isnan(perna)) else np.nan)
                res[f'Torn_{lado}'].append(pe - perna if not (np.isnan(pe) or np.isnan(perna)) else np.nan)
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
            
            curve_hs, curve_to = cal_x_data[0] - pelvis_x, toe_x_data[0] - pelvis_x
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

    def _calcular_tempo_medio_ciclo(self):
        tempos = []
        for lado in ['D', 'E']:
            hss = self.eventos[lado]['HS']
            if len(hss) > 1:
                for i in range(len(hss) - 1):
                    tempos.append((hss[i+1] - hss[i]) / self.freq)
        return np.mean(tempos) if tempos else np.nan

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
                    angulos = np.mod(np.degrees(np.arctan2(np.diff(cd), np.diff(cp))), 360)
                    counts = {'Proximal': 0, 'Distal': 0, 'EmFase': 0, 'AntiFase': 0}
                    for a in angulos:
                        if (0 <= a < 22.5) or (337.5 <= a <= 360) or (157.5 <= a < 202.5): counts['Proximal'] += 1
                        elif (22.5 <= a < 67.5) or (202.5 <= a < 247.5): counts['EmFase'] += 1
                        elif (67.5 <= a < 112.5) or (247.5 <= a < 292.5): counts['Distal'] += 1
                        else: counts['AntiFase'] += 1
                    for k in counts: freqs[k].append((counts[k] / len(angulos)) * 100)
                
                for k in freqs: res[nome_par][k] = np.mean(freqs[k])
                
                c_prox_m, c_dist_m = np.mean(c_prox, axis=0), np.mean(c_dist, axis=0)
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
            
        for l, cal, t1, t5, ank in [('D', get('RCAL'), get('RFT1'), get('RFT5'), td), ('E', get('LCAL'), get('LFT1'), get('LFT5'), te)]:
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

        ax_stats_left = fig.add_axes([0.01, 0.02, 0.30, 0.32])
        ax_stats_left.axis('off')
        
        ax = fig.add_axes([0.32, 0.20, 0.44, 0.75], projection='3d')
        ax.set_xlim(self.box['x']); ax.set_ylim(self.box['y']); ax.set_zlim(self.box['z'])
        ax.view_init(elev=20, azim=135)
        ax.set_xlabel('X (Inv)'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        titulo_main = ax.set_title(self.nome_arq, fontsize=12, pad=20)

        ax_banner = fig.add_axes([0.33, 0.02, 0.43, 0.16]); ax_banner.axis('off')
        ax_txt = fig.add_axes([0.78, 0.05, 0.21, 0.90]); ax_txt.axis('off')

        stats_ang = self.proc.obter_stats()
        coord_norm = self.proc.coord_vetorial

        ax_stats_left.text(0.5, 1.0, "FREQUÊNCIA NO CICLO DA MARCHA (0-100%)", ha='center', va='top', fontweight='bold', fontsize=10)
        def format_f(c): return f" Proximal : {c.get('Proximal',0):>3.0f}%\n Em Fase  : {c.get('EmFase',0):>3.0f}%\n Distal   : {c.get('Distal',0):>3.0f}%\n Anti-Fase: {c.get('AntiFase',0):>3.0f}%"
        col_dir = ">> COXA-PERNA (DIR)\n" + format_f(coord_norm.get('Coxa_Perna_D', {})) + "\n\n>> PERNA-PÉ (DIR)\n" + format_f(coord_norm.get('Perna_Pe_D', {}))
        col_esq = ">> COXA-PERNA (ESQ)\n" + format_f(coord_norm.get('Coxa_Perna_E', {})) + "\n\n>> PERNA-PÉ (ESQ)\n" + format_f(coord_norm.get('Perna_Pe_E', {}))
        ax_stats_left.text(0.00, 0.85, col_dir, va='top', fontsize=9, family='monospace')
        ax_stats_left.text(0.55, 0.85, col_esq, va='top', fontsize=9, family='monospace')

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
                if k not in seg: 
                    linhas[k].remove()
                    del linhas[k]
            for n, (p1, p2) in seg.items():
                c = 'red' if 'D' in n or 'R' in n else 'blue'
                if 'P_' in n or 'PL' in n or 'PR' in n: c = 'black'
                if n in linhas:
                    linhas[n].set_data([p1[0],p2[0]],[p1[1],p2[1]])
                    linhas[n].set_3d_properties([p1[2],p2[2]])
                else: 
                    linhas[n], = ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]], c=c, lw=2)

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
                        ptr.set_data(
