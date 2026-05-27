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
from statsmodels.multivariate.manova import MANOVA

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(page_title="GPBIO - Biomecânica Clínica", layout="wide", page_icon="🚶")

# =============================================================================
# ENGINE MATEMÁTICA E SEGMENTAR
# =============================================================================
def normalizar(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

# [NOVO] Função para projetar vetores no Plano Sagital (X e Z)
def vetor_sagital(p1, p2): 
    if p1 is None or p2 is None: 
        return None
    # Considerando X (índice 0) como eixo de progressão e Z (índice 2) como eixo vertical.
    # Ignora-se o eixo Y (índice 1 - Médio-lateral)
    return np.array([p2[0] - p1[0], p2[2] - p1[2]])

def angulo_entre(v1, v2):
    if v1 is None or v2 is None: 
        return np.nan
    v1_u, v2_u = normalizar(v1), normalizar(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

class ProcessadorCinematico:
    def __init__(self, caminho_arquivo, nome_original, grupo="Geral", df_antropo=None):
        self.caminho = caminho_arquivo
        self.nome_arq = nome_original
        self.grupo = grupo
        self.valido = False
        self.erro_msg = ""
        
        # [NOVO] Extração do ID do paciente baseada no prefixo do arquivo
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
            self.angulos_df = self._calcular_angulos() 
            self.segmentos_df = self._calcular_angulos_segmentares() 
            
            self.velocidade_media = self._calcular_velocidade_sacrum()
            self.eventos = self.detectar_eventos_zeni()
            self.fases_marcha = self._calcular_fases_marcha()
            self.foot_clearance = self._calcular_foot_clearance()
            self.comprimento_passo = self._calcular_comprimento_passo()

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

    # [ALTERADO] Cálculo Articular agora utiliza projeção 2D (Sagital)
    def _calcular_angulos(self):
        res = {k: [] for k in ['Quad_D','Joel_D','Torn_D','Quad_E','Joel_E','Torn_E']}
        vec_g_2d = np.array([0, -1]) # Vetor gravidade projetado em 2D (apontando para baixo em Z)
        
        for f in range(self.n_frames):
            # Lado Direito
            h_d = self._get('RIAS',f); k_d = self._mid('RLE','RME',f); a_d = self._mid('RML','RMM',f); p_d = self._mid('RFT1','RFT5',f)
            v_coxa_d = vetor_sagital(h_d, k_d)
            v_perna_d = vetor_sagital(k_d, a_d)
            v_pe_d = vetor_sagital(a_d, p_d)
            
            res['Quad_D'].append(angulo_entre(v_coxa_d, vec_g_2d))
            res['Joel_D'].append(angulo_entre(v_coxa_d, v_perna_d))
            res['Torn_D'].append(angulo_entre(v_perna_d, v_pe_d))
            
            # Lado Esquerdo
            h_e = self._get('LIAS',f); k_e = self._mid('LLE','LME',f); a_e = self._mid('LML','LMM',f); p_e = self._mid('LFT1','LFT5',f)
            v_coxa_e = vetor_sagital(h_e, k_e)
            v_perna_e = vetor_sagital(k_e, a_e)
            v_pe_e = vetor_sagital(a_e, p_e)
            
            res['Quad_E'].append(angulo_entre(v_coxa_e, vec_g_2d))
            res['Joel_E'].append(angulo_entre(v_coxa_e, v_perna_e))
            res['Torn_E'].append(angulo_entre(v_perna_e, v_pe_e))
            
        return pd.DataFrame(res)

    # [ALTERADO] Cálculo Segmentar agora utiliza projeção 2D (Sagital)
    def _calcular_angulos_segmentares(self):
        res = {k: [] for k in ['Coxa_D','Perna_D','Pe_D','Coxa_E','Perna_E','Pe_E']}
        vec_v_2d = np.array([0, -1]) # Vertical em 2D
        vec_h_2d = np.array([1, 0])  # Horizontal em 2D
        
        for f in range(self.n_frames):
            for lado, l in [('D', 'R'), ('E', 'L')]:
                h = self._get(f'{l}IAS',f)
                k = self._mid(f'{l}LE', f'{l}ME',f)
                a = self._mid(f'{l}ML', f'{l}MM',f)
                p = self._mid(f'{l}FT1', f'{l}FT5',f)
                cal = self._get(f'{l}CAL',f)
                
                v_coxa = vetor_sagital(h, k)
                v_perna = vetor_sagital(k, a)
                v_pe = vetor_sagital(cal, p)
                
                res[f'Coxa_{lado}'].append(angulo_entre(v_coxa, vec_v_2d))
                res[f'Perna_{lado}'].append(angulo_entre(v_perna, vec_v_2d))
                res[f'Pe_{lado}'].append(angulo_entre(v_pe, vec_h_2d))
                
        return pd.DataFrame(res)

    def detectar_eventos_zeni(self):
        eventos = {'D': {'HS': [], 'TO': []}, 'E': {'HS': [], 'TO': []}}
        rias_data = self._get('RIAS', slice(None))
        lias_data = self._get('LIAS', slice(None))
        
        if rias_data is None or lias_data is None: 
            return eventos 
            
        pelvis_x = (rias_data[0] + lias_data[0]) / 2
        dist_frames = int(self.freq * 0.6)
        
        for lado, cal_label, toe_label in [('D','RCAL','RFT1'), ('E','LCAL','LFT1')]:
            cal_x_data = self._get(cal_label, slice(None))
            toe_x_data = self._get(toe_label, slice(None))
            
            if cal_x_data is None or toe_x_data is None: 
                continue
            
            curve_hs = cal_x_data[0] - pelvis_x
            curve_to = toe_x_data[0] - pelvis_x
            
            if np.nanmean(curve_hs) > 0:
                picos_hs, _ = signal.find_peaks(-curve_hs, distance=dist_frames)
                vales_to, _ = signal.find_peaks(curve_to, distance=dist_frames)
            else:
                picos_hs, _ = signal.find_peaks(curve_hs, distance=dist_frames)
                vales_to, _ = signal.find_peaks(-curve_to, distance=dist_frames)
                
            eventos[lado]['HS'], eventos[lado]['TO'] = sorted(picos_hs), sorted(vales_to)
            
        return eventos

    def obter_stats(self):
        if not self.valido: 
            return None
        return {col: {'min': self.angulos_df[col].min(), 'max': self.angulos_df[col].max()} for col in self.angulos_df.columns}

    def _calcular_velocidade_sacrum(self):
        rips, lips = self._get('RIPS', slice(None)), self._get('LIPS', slice(None))
        if rips is None or lips is None: 
            return np.nan
        sacrum = (rips + lips) / 2
        return np.nanmean(np.linalg.norm(np.diff(sacrum, axis=1), axis=0) * self.freq / 1000.0)

    def _calcular_fases_marcha(self):
        res = {'D': {'Apoio': np.nan, 'Balanco': np.nan}, 'E': {'Apoio': np.nan, 'Balanco': np.nan}}
        for lado in ['D', 'E']:
            hss, tos = self.eventos[lado]['HS'], self.eventos[lado]['TO']
            if len(hss) < 2 or not tos: 
                continue
            ciclos_apoio = []
            for i in range(len(hss) - 1):
                to_valido = [t for t in tos if hss[i] < t < hss[i+1]]
                if to_valido:
                    pct = ((to_valido[0] - hss[i]) / (hss[i+1] - hss[i])) * 100
                    if pct < 45.0: 
                        pct = 100.0 - pct
                    ciclos_apoio.append(pct)
            if ciclos_apoio: 
                res[lado]['Apoio'] = np.mean(ciclos_apoio)
                res[lado]['Balanco'] = 100.0 - np.mean(ciclos_apoio)
        return res

    def _calcular_foot_clearance(self):
        res = {'D': np.nan, 'E': np.nan}
        for lado, pref in [('D', 'R'), ('E', 'L')]:
            ft1 = self._get(f'{pref}FT1', slice(None))
            if ft1 is None: 
                continue
            hss, tos = self.eventos[lado]['HS'], self.eventos[lado]['TO']
            alturas = [np.max(ft1[2, to_f:min([h for h in hss if h > to_f] or [to_f])]) for to_f in tos if [h for h in hss if h > to_f]]
            if alturas: 
                res[lado] = np.mean(alturas)
        return res

    def _calcular_comprimento_passo(self):
        res = {'D': np.nan, 'E': np.nan}
        rcal, lcal = self._get('RCAL', slice(None)), self._get('LCAL', slice(None))
        rft1, lft1 = self._get('RFT1', slice(None)), self._get('LFT1', slice(None))
        if any(m is None for m in [rcal, lcal, rft1, lft1]): 
            return res
        passos_d = [np.linalg.norm(rcal[0:2, hs] - lft1[0:2, hs]) for hs in self.eventos['D']['HS'] if hs < rcal.shape[1] and hs < lft1.shape[1]]
        passos_e = [np.linalg.norm(lcal[0:2, hs] - rft1[0:2, hs]) for hs in self.eventos['E']['HS'] if hs < lcal.shape[1] and hs < rft1.shape[1]]
        if passos_d: 
            res['D'] = np.mean(passos_d)
        if passos_e: 
            res['E'] = np.mean(passos_e)
        return res

    def extrair_ciclos_normalizados(self, vetor_dados, eventos_hs, pontos=101):
        ciclos = []
        if len(eventos_hs) < 2: 
            return []
        for i in range(len(eventos_hs) - 1):
            if eventos_hs[i+1] > len(vetor_dados): 
                continue
            ciclo_bruto = vetor_dados[eventos_hs[i]:eventos_hs[i+1]]
            ciclos.append(np.interp(np.linspace(0, len(ciclo_bruto)-1, pontos), np.arange(len(ciclo_bruto)), ciclo_bruto))
        return ciclos

    def _calcular_coordenacao_vetorial(self):
        res = {}
        self.coord_vetorial_series = {} 
        
        for lado in ['D', 'E']:
            hss = self.eventos[lado]['HS']
            if len(hss) < 2: 
                continue
            
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
                if not c_prox or not c_dist: 
                    continue
                
                res[nome_par] = {'Proximal': np.nan, 'Distal': np.nan, 'EmFase': np.nan, 'AntiFase': np.nan}
                freqs = {'Proximal': [], 'Distal': [], 'EmFase': [], 'AntiFase': []}
                
                for cp, cd in zip(c_prox, c_dist):
                    angulos = np.mod(np.degrees(np.arctan2(np.diff(cd), np.diff(cp))), 360)
                    counts = {'Proximal': 0, 'Distal': 0, 'EmFase': 0, 'AntiFase': 0}
                    for a in angulos:
                        if (0 <= a < 22.5) or (337.5 <= a <= 360) or (157.5 <= a < 202.5): 
                            counts['Proximal'] += 1
                        elif (22.5 <= a < 67.5) or (202.5 <= a < 247.5): 
                            counts['EmFase'] += 1
                        elif (67.5 <= a < 112.5) or (247.5 <= a < 292.5): 
                            counts['Distal'] += 1
                        else: 
                            counts['AntiFase'] += 1
                    for k in counts: 
                        freqs[k].append((counts[k] / len(angulos)) * 100)
                
                for k in freqs: 
                    res[nome_par][k] = np.mean(freqs[k])
                
                c_prox_m = np.mean(c_prox, axis=0)
                c_dist_m = np.mean(c_dist, axis=0)
                ang_m = np.mod(np.degrees(np.arctan2(np.diff(c_dist_m), np.diff(c_prox_m))), 360)
                fatia_media = []
                for a in ang_m:
                    if (0 <= a < 22.5) or (337.5 <= a <= 360) or (157.5 <= a < 202.5): 
                        fatia_media.append('Proximal')
                    elif (22.5 <= a < 67.5) or (202.5 <= a < 247.5): 
                        fatia_media.append('EmFase')
                    elif (67.5 <= a < 112.5) or (247.5 <= a < 292.5): 
                        fatia_media.append('Distal')
                    else: 
                        fatia_media.append('AntiFase')
                
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
        
        if rias is not None and lias is not None: 
            s['P_F']=[rias,lias]
        if rips is not None and lips is not None: 
            s['P_B']=[rips,lips]
        if rias is not None and rict is not None: 
            s['PR1']=[rias,rict]
        if rips is not None and rict is not None: 
            s['PR2']=[rips,rict] 
        if lias is not None and lict is not None: 
            s['PL1']=[lias,lict]
        if lips is not None and lict is not None: 
            s['PL2']=[lips,lict] 
        
        kd, ke = mid('RLE','RME'), mid('LLE','LME')
        td, te = mid('RML','RMM'), mid('LML','LMM')
        if rias is not None and kd is not None: 
            s['CX_D']=[rias,kd]
        if lias is not None and ke is not None: 
            s['CX_E']=[lias,ke]
        if kd is not None and td is not None: 
            s['PN_D']=[kd,td]
        if ke is not None and te is not None: 
            s['PN_E']=[ke,te]
            
        for l, cal, t1, t5, ank in [('D', get('RCAL'), get('RFT1'), get('RFT5'), td),
                                    ('E', get('LCAL'), get('LFT1'), get('LFT5'), te)]:
            if cal is not None and t1 is not None: 
                s[f'P{l}1']=[cal,t1]
            if cal is not None and t5 is not None: 
                s[f'P{l}2']=[cal,t5]
            if t1 is not None and t5 is not None: 
                s[f'P{l}3']=[t1,t5]
            if ank is not None and cal is not None: 
                s[f'P{l}L']=[ank,cal]
        return s

    def _desenhar_fundo_bussola(self, ax_c, titulo):
        ax_c.set_xlim(-1.2, 1.2)
        ax_c.set_ylim(-1.2, 1.2)
        ax_c.axis('off')
        ax_c.set_aspect('equal')
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
        if np.isnan(angulo): 
            return "-", "gray"
        a = angulo % 360
        if (0 <= a < 22.5) or (337.5 <= a <= 360) or (157.5 <= a < 202.5): 
            return "PROXIMAL", '#e74c3c'
        elif (22.5 <= a < 67.5) or (202.5 <= a < 247.5): 
            return "EM FASE", '#2ecc71'
        elif (67.5 <= a < 112.5) or (247.5 <= a < 292.5): 
            return "DISTAL", '#3498db'
        else: 
            return "ANTI-FASE", '#f1c40f'

    def salvar(self, caminho_final, step=3, fps_anim=20):
        fig = plt.figure(figsize=(16, 9))
        
        ax_comp_qj_d = fig.add_axes([0.01, 0.65, 0.15, 0.25])
        ax_comp_jt_d = fig.add_axes([0.01, 0.38, 0.15, 0.25])
        ax_comp_qj_e = fig.add_axes([0.16, 0.65, 0.15, 0.25])
        ax_comp_jt_e = fig.add_axes([0.16, 0.38, 0.15, 0.25])

        ptr_qjd = self._desenhar_fundo_bussola(ax_comp_qj_d, "Coxa-Perna (DIR)")
        ptr_jtd = self._desenhar_fundo_bussola(ax_comp_jt_d, "Perna-Pé (DIR)")
        ptr_qje = self._desenhar_fundo_bussola(ax_comp_qj_e, "Coxa-Perna (ESQ)")
        ptr_jte = self._desenhar_fundo_bussola(ax_comp_jt_e, "Perna-Pé (ESQ)")

        ax_stats_left = fig.add_axes([0.01, 0.02, 0.30, 0.32])
        ax_stats_left.axis('off')
        
        ax = fig.add_axes([0.32, 0.20, 0.44, 0.75], projection='3d')
        ax.set_xlim(self.box['x'])
        ax.set_ylim(self.box['y'])
        ax.set_zlim(self.box['z'])
        ax.view_init(elev=20, azim=135)
        ax.set_xlabel('X (Inv)')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        titulo_main = ax.set_title(self.nome_arq, fontsize=12, pad=20)

        ax_banner = fig.add_axes([0.33, 0.02, 0.43, 0.16])
        ax_banner.axis('off')
        ax_txt = fig.add_axes([0.78, 0.05, 0.21, 0.90])
        ax_txt.axis('off')

        stats_ang = self.proc.obter_stats()
        coord_norm = self.proc.coord_vetorial

        ax_stats_left.text(0.5, 1.0, "FREQUÊNCIA NO CICLO DA MARCHA (0-100%)", ha='center', va='top', fontweight='bold', fontsize=10)
        def format_f(c): 
            return f" Proximal : {c.get('Proximal',0):>3.0f}%\n Em Fase  : {c.get('EmFase',0):>3.0f}%\n Distal   : {c.get('Distal',0):>3.0f}%\n Anti-Fase: {c.get('AntiFase',0):>3.0f}%"
        
        col_dir = ">> COXA-PERNA (DIR)\n" + format_f(coord_norm.get('Coxa_Perna_D', {})) + "\n\n"
        col_dir += ">> PERNA-PÉ (DIR)\n" + format_f(coord_norm.get('Perna_Pe_D', {}))
        
        col_esq = ">> COXA-PERNA (ESQ)\n" + format_f(coord_norm.get('Coxa_Perna_E', {})) + "\n\n"
        col_esq += ">> PERNA-PÉ (ESQ)\n" + format_f(coord_norm.get('Perna_Pe_E', {}))

        ax_stats_left.text(0.00, 0.85, col_dir, va='top', fontsize=9, family='monospace')
        ax_stats_left.text(0.55, 0.85, col_esq, va='top', fontsize=9, family='monospace')

        ax_banner.text(0.5, 0.90, "COORDENAÇÃO SEGMENTAR EM TEMPO REAL", ha='center', va='top', fontweight='bold', fontsize=11)
        
        ax_banner.text(0.00, 0.50, "Coxa-Perna (DIR):", fontweight='bold', fontsize=10)
        ax_banner.text(0.00, 0.15, "Perna-Pé (DIR):", fontweight='bold', fontsize=10)
        txt_qj_d = ax_banner.text(0.24, 0.50, "-", fontweight='bold', fontsize=10)
        txt_jt_d = ax_banner.text(0.24, 0.15, "-", fontweight='bold', fontsize=10)

        ax_banner.text(0.53, 0.50, "Coxa-Perna (ESQ):", fontweight='bold', fontsize=10)
        ax_banner.text(0.53, 0.15, "Perna-Pé (ESQ):", fontweight='bold', fontsize=10)
        txt_qj_e = ax_banner.text(0.77, 0.50, "-", fontweight='bold', fontsize=10)
        txt_jt_e = ax_banner.text(0.77, 0.15, "-", fontweight='bold', fontsize=10)

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
                if 'P_' in n or 'PL' in n or 'PR' in n: 
                    c = 'black'
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
                p_prox = self.proc.segmentos_df.iloc[i+1]
                p_curr = self.proc.segmentos_df.iloc[i]
                
                pares = [
                    ('Coxa_D', 'Perna_D', ptr_qjd, txt_qj_d),
                    ('Perna_D', 'Pe_D', ptr_jtd, txt_jt_d),
                    ('Coxa_E', 'Perna_E', ptr_qje, txt_qj_e),
                    ('Perna_E', 'Pe_E', ptr_jte, txt_jt_e)
                ]

                for j_prox, j_dist, ptr, txt in pares:
                    dx = p_prox[j_prox] - p_curr[j_prox]
                    dy = p_prox[j_dist] - p_curr[j_dist]
                    ang = np.degrees(np.arctan2(dy, dx)) % 360 if not (np.isnan(dx) or np.isnan(dy)) else np.nan
                    
                    if not np.isnan(ang):
                        ptr.set_data([0, np.cos(np.radians(ang))], [0, np.sin(np.radians(ang))])
                        label, cor = self._classificar_angulo(ang)
                        txt.set_text(label)
                        txt.set_color(cor)

            return list(linhas.values()) + [t_dynamic, txt_qj_d, txt_jt_d, txt_qj_e, txt_jt_e, ptr_qjd, ptr_jtd, ptr_qje, ptr_jte]

        ani = animation.FuncAnimation(fig, update, frames=range(0, self.proc.n_frames, step), interval=50)
        try:
            ani.save(caminho_final, writer='pillow', fps=fps_anim)
            return True, caminho_final
        except Exception as e: 
            return False, str(e)
        finally: 
            plt.close(fig)
            plt.close('all')

# =============================================================================
# INTERFACE WEB STREAMLIT
# =============================================================================
st.title("🚶 GPBIO - Sistema de Análise de Marcha")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📏 Dados Antropométricos")
st.sidebar.info("Planilha com colunas 'ID' e 'Altura' (em metros). O ID deve bater com as iniciais do arquivo C3D.")
arquivo_antropo = st.sidebar.file_uploader("Planilha de Altura (Excel/CSV)", type=['xlsx', 'csv'])

df_antropo = None
if arquivo_antropo:
    if arquivo_antropo.name.endswith('xlsx'): 
        df_antropo = pd.read_excel(arquivo_antropo)
    else: 
        df_antropo = pd.read_csv(arquivo_antropo, sep=';', decimal=',')
    
    df_antropo.columns = df_antropo.columns.str.strip().str.upper()
    
    if 'ID' in df_antropo.columns and 'ALTURA' in df_antropo.columns:
        df_antropo['ID'] = df_antropo['ID'].astype(str).str.upper().str.strip()
        st.sidebar.success(f"Dados de {len(df_antropo)} participantes prontos para normalização!")
    else:
        st.sidebar.error("A planilha precisa ter as colunas 'ID' e 'ALTURA'. Encontrado: " + ", ".join(df_antropo.columns))
        df_antropo = None

st.sidebar.markdown("---")
st.sidebar.markdown("### Sobre o Sistema")
st.sidebar.info("GPBIO: Análise Biomecânica de Marcha.")
st.sidebar.markdown("**Desenvolvido por Arthur Lins**")
	
if 'processadores' not in st.session_state: 
    st.session_state.processadores = []

st.subheader("📁 Importação de Dados e Separação de Grupos")
st.info("Digite os nomes dos grupos do seu estudo e faça o upload dos arquivos .c3d dinâmicos em suas respectivas áreas.")

col_g1, col_g2 = st.columns(2)
with col_g1:
    nome_g1 = st.text_input("Nome do Grupo 1 (Ex: Controle)", value="Controle")
    files_g1 = st.file_uploader(f"Arquivos C3D - {nome_g1}", type=['c3d'], accept_multiple_files=True, key="up_g1")
with col_g2:
    nome_g2 = st.text_input("Nome do Grupo 2 (Ex: Parkinson)", value="Intervenção")
    files_g2 = st.file_uploader(f"Arquivos C3D - {nome_g2}", type=['c3d'], accept_multiple_files=True, key="up_g2")

if st.button("Processar e Agrupar Arquivos", type="primary", use_container_width=True):
    arquivos_para_processar = []
    if files_g1: 
        arquivos_para_processar.extend([(f, nome_g1) for f in files_g1 if "CAL" not in f.name.upper()])
    if files_g2: 
        arquivos_para_processar.extend([(f, nome_g2) for f in files_g2 if "CAL" not in f.name.upper()])

    if not arquivos_para_processar:
        st.warning("Faça o upload de arquivos dinâmicos em pelo menos um dos grupos para continuar.")
    else:
        st.session_state.processadores = []
        progress_bar = st.progress(0)
        
        for i, (file, nome_grupo) in enumerate(arquivos_para_processar):
            file.seek(0) 
            with tempfile.NamedTemporaryFile(delete=False, suffix='.c3d') as tmp_file:
                tmp_file.write(file.read()) 
                tmp_file.flush()            
                os.fsync(tmp_file.fileno()) 
                tmp_path = tmp_file.name
                
            proc = ProcessadorCinematico(tmp_path, file.name, grupo=nome_grupo, df_antropo=df_antropo)
            if proc.valido: 
                st.session_state.processadores.append(proc)
            else: 
                st.error(f"Erro no arquivo {file.name}: {proc.erro_msg}")
                
            try: 
                os.remove(tmp_path)
            except Exception: 
                pass 
                
            progress_bar.progress((i + 1) / len(arquivos_para_processar))
            
        st.success(f"✅ {len(st.session_state.processadores)} arquivos processados e agrupados com sucesso!")

if st.session_state.processadores:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Tabela de Médias", "📈 Gráficos de Curvas", "⚙️ Coordenação (Angle-Angle)", 
        "🎥 Animações 3D (GIFs)", "📦 Estatística (Boxplots e Barras)", "🧪 Testes de Hipótese", "📝 Relatório Clínico"
    ])

    with tab1:
        st.subheader("📊 Tabela de Dados Agrupados (Média por Paciente)")
        st.write("Os dados abaixo representam a média de todas as tentativas processadas para cada paciente.")
        
        dados_tabela = []
        for p in st.session_state.processadores:
            try:
                # [ALTERADO] Inclusão do ID_Paciente para permitir o agrupamento (groupby)
                linha = {
                    "ID_Paciente": p.id_paciente,
                    "Grupo": p.grupo,
                    "Velocidade (m/s)": getattr(p, 'velocidade_media', np.nan),
                    "Apoio DIR (%)": p.fases_marcha.get('D', {}).get('Apoio', np.nan),
                    "Apoio ESQ (%)": p.fases_marcha.get('E', {}).get('Apoio', np.nan),
                    "Clearance DIR (mm)": p.foot_clearance.get('D', np.nan),
                    "Clearance ESQ (mm)": p.foot_clearance.get('E', np.nan),
                    "Passo DIR (mm)": p.comprimento_passo.get('D', np.nan),
                    "Passo ESQ (mm)": p.comprimento_passo.get('E', np.nan),
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
                                linha[f"CAV {par_label} (°)"] = np.nan
                                linha[f"Transições {par_label}"] = np.nan
                        else:
                            linha[f"CAV {par_label} (°)"] = np.nan
                            linha[f"Transições {par_label}"] = np.nan

                        serie = p.coord_vetorial_series.get(par_key, [])
                        fatia_apoio = serie[0:60] if len(serie) >= 60 else []
                        fatia_balanco = serie[60:] if len(serie) > 60 else []

                        for padrao in padroes:
                            linha[f"APOIO {par_label} - {padrao} (%)"] = (fatia_apoio.count(padrao) / len(fatia_apoio)) * 100 if len(fatia_apoio) > 0 else np.nan
                            linha[f"BALANÇO {par_label} - {padrao} (%)"] = (fatia_balanco.count(padrao) / len(fatia_balanco)) * 100 if len(fatia_balanco) > 0 else np.nan
                    except Exception:
                        linha[f"CAV {par_label} (°)"] = np.nan
                        linha[f"Transições {par_label}"] = np.nan
                
                # Armazenar tudo como float puro no primeiro momento para o pandas fazer o agrupamento com precisão
                dados_tabela.append(linha)
            except Exception: 
                continue
                
        if dados_tabela:
            # [NOVA LÓGICA DE AGREGAÇÃO] Agrupar as tentativas por paciente calculando a média
            df_bruto = pd.DataFrame(dados_tabela)
            cols_num = df_bruto.select_dtypes(include=[np.number]).columns.tolist()
            
            # Agrupa calculando a média das tentativas para cada participante individualmente
            df_agrupado_pacientes = df_bruto.groupby(['Grupo', 'ID_Paciente'])[cols_num].mean().reset_index()
            df_agrupado_pacientes = df_agrupado_pacientes.round(2)
            
            # Formata NaNs como string vazia para display limpo
            df_agrupado_pacientes = df_agrupado_pacientes.replace(np.nan, "")

            # Exibir na Interface
            st.dataframe(df_agrupado_pacientes, use_container_width=True, height=600)
            
            # Exportar o CSV do DataFrame já agrupado
            csv = df_agrupado_pacientes.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
            st.download_button("📥 Baixar Tabela Agrupada por Paciente (CSV)", data=csv, file_name="estatistica_agrupada_participantes.csv", mime="text/csv", type="primary")
        else: 
            st.info("Importe arquivos na barra lateral.")

    with tab2:
        st.subheader("Cinemática Articular Normalizada (Média dos Grupos)")
        grupos_estudo = sorted(list(set([p.grupo for p in st.session_state.processadores])))
        cols_t2 = st.columns(len(grupos_estudo))

        for idx, grp in enumerate(grupos_estudo):
            with cols_t2[idx]:
                st.markdown(f"<h4 style='text-align:center; color: #555;'>Grupo: {grp}</h4>", unsafe_allow_html=True)
                procs_grp = [p for p in st.session_state.processadores if p.grupo == grp]
                dados_grp = { 'Quad_D': [], 'Joel_D': [], 'Torn_D': [], 'Quad_E': [], 'Joel_E': [], 'Torn_E': [] }
                
                for proc in procs_grp:
                    for joint in ['Quad', 'Joel', 'Torn']:
                        for lado in ['D', 'E']:
                            chave = f"{joint}_{lado}"
                            hss = proc.eventos[lado]['HS']
                            ciclos = proc.extrair_ciclos_normalizados(proc.angulos_df[chave].values, hss)
                            dados_grp[chave].extend(ciclos)

                fig, axs = plt.subplots(3, 2, figsize=(7, 9), sharex=True)
                mapeamento = [('Quad_D', 0, 0, 'Quad (DIR)'), ('Quad_E', 0, 1, 'Quad (ESQ)'), 
                              ('Joel_D', 1, 0, 'Joel (DIR)'), ('Joel_E', 1, 1, 'Joel (ESQ)'), 
                              ('Torn_D', 2, 0, 'Torn (DIR)'), ('Torn_E', 2, 1, 'Torn (ESQ)')]
                x_axis = np.linspace(0, 100, 101)

                for chave, row, col, titulo in mapeamento:
                    ax = axs[row, col]
                    ciclos = np.array(dados_grp[chave])
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.set_title(titulo, fontweight='bold', fontsize=10)
                    if col == 0: 
                        ax.set_ylabel("Graus (°)", fontsize=9)
                    if row == 2: 
                        ax.set_xlabel("% Ciclo", fontsize=9)

                    if len(ciclos) > 0:
                        media, std = np.mean(ciclos, axis=0), np.std(ciclos, axis=0)
                        ax.fill_between(x_axis, media - std, media + std, color='gray', alpha=0.3)
                        ax.plot(x_axis, media, color='blue' if col==1 else 'red', lw=2)
                        ax.axhline(0, color='black', lw=0.8)
                    else: 
                        ax.text(50, 0, "Sem Dados", ha='center')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    with tab3:
        st.subheader("⚙️ Coordenação Vetorial e Controle Motor")
        tipo_coord = st.radio("Selecione o Modelo de Análise Biomecânica:", 
                              ["📐 Coordenação Segmentar (Absoluta - Padrão Ouro)", "📍 Coordenação Articular (Relativa)"], 
                              horizontal=True)
        
        if "Segmentar" in tipo_coord:
            pares_map = [('Coxa_Perna_D', 'Coxa-Perna (DIR)'), ('Coxa_Perna_E', 'Coxa-Perna (ESQ)'), ('Perna_Pe_D', 'Perna-Pé (DIR)'), ('Perna_Pe_E', 'Perna-Pé (ESQ)')]
            label_prox_1, label_dist_1 = 'Coxa', 'Perna'
            label_prox_2, label_dist_2 = 'Perna', 'Pe'
            label_cav_1, label_cav_2 = 'Coxa_Perna', 'Perna_Pe'
            df_ref = 'segmentos_df'
        else:
            pares_map = [('Quad_Joel_D', 'Quad-Joel (DIR)'), ('Quad_Joel_E', 'Quad-Joel (ESQ)'), ('Joel_Torn_D', 'Joel-Torn (DIR)'), ('Joel_Torn_E', 'Joel-Torn (ESQ)')]
            label_prox_1, label_dist_1 = 'Quad', 'Joel'
            label_prox_2, label_dist_2 = 'Joel', 'Torn'
            label_cav_1, label_cav_2 = 'Quad_Joel', 'Joel_Torn'
            df_ref = 'angulos_df'

        st.markdown("---")
        grupos_estudo = sorted(list(set([p.grupo for p in st.session_state.processadores])))
        cor_g1, cor_g2 = '#a8c8f9', '#f9a8a8'
        padroes = {'Proximal': '#e74c3c', 'EmFase': '#2ecc71', 'Distal': '#3498db', 'AntiFase': '#f1c40f'}

        freq_acumulada = {g: {c_new: {k: [] for k in padroes} for _, c_new in pares_map} for g in grupos_estudo}
        cav_data = {label_cav_1: {g: [] for g in grupos_estudo}, label_cav_2: {g: [] for g in grupos_estudo}}
        trans_data = {label_cav_1: {g: [] for g in grupos_estudo}, label_cav_2: {g: [] for g in grupos_estudo}}

        for p in st.session_state.processadores:
            grp = p.grupo
            for c_old, c_new in pares_map:
                freqs = p.coord_vetorial.get(c_old, {})
                for k in padroes.keys():
                    if not np.isnan(freqs.get(k, np.nan)): 
                        freq_acumulada[grp][c_new][k].append(freqs[k])

            for prox, dist, label_cav in [(label_prox_1, label_dist_1, label_cav_1), (label_prox_2, label_dist_2, label_cav_2)]:
                for lado in ['D', 'E']:
                    chave_prox, chave_dist = f"{prox}_{lado}", f"{dist}_{lado}"
                    hss = p.eventos[lado]['HS']
                    if len(hss) > 1:
                        df_obj = getattr(p, df_ref)
                        c_prox = p.extrair_ciclos_normalizados(df_obj[chave_prox].values, hss)
                        c_dist = p.extrair_ciclos_normalizados(df_obj[chave_dist].values, hss)
                        
                        if len(c_prox) > 0 and len(c_dist) > 0:
                            arr_p, arr_d = np.array(c_prox), np.array(c_dist)
                            delta_p, delta_d = np.diff(arr_p, axis=1), np.diff(arr_d, axis=1)
                            gamma_rad = np.arctan2(delta_d, delta_p)
                            gamma_deg = (np.degrees(gamma_rad) + 360) % 360
                            
                            padroes_idx = np.digitize(gamma_deg, [0, 45, 135, 225, 315, 360])
                            padroes_idx[padroes_idx == 5] = 1 
                            trans_data[label_cav][grp].append(np.mean(np.sum(np.diff(padroes_idx, axis=1) != 0, axis=1)))
                            
                            x_m, y_m = np.mean(np.cos(gamma_rad), axis=0), np.mean(np.sin(gamma_rad), axis=0)
                            cav_data[label_cav][grp].append(np.mean(np.sqrt(2 * (1 - np.clip(np.sqrt(x_m**2 + y_m**2), 0, 1))) * (180 / np.pi)))

        st.markdown("### 1. Comportamento Espacial (Diagramas Angle-Angle)")
        cols_t3 = st.columns(len(grupos_estudo))

        for idx, grp in enumerate(grupos_estudo):
            with cols_t3[idx]:
                st.markdown(f"<h4 style='text-align:center; color: #555;'>Grupo: {grp}</h4>", unsafe_allow_html=True)
                procs_grp = [p for p in st.session_state.processadores if p.grupo == grp]
                dados_grp = { f'{label_prox_1}_D': [], f'{label_dist_1}_D': [], f'{label_dist_2}_D': [], f'{label_prox_1}_E': [], f'{label_dist_1}_E': [], f'{label_dist_2}_E': [] }
                
                for proc in procs_grp:
                    for joint in [label_prox_1, label_dist_1, label_dist_2]:
                        for lado in ['D', 'E']:
                            chave = f"{joint}_{lado}"
                            df_obj = getattr(proc, df_ref)
                            ciclos = proc.extrair_ciclos_normalizados(df_obj[chave].values, proc.eventos[lado]['HS'])
                            dados_grp[chave].extend(ciclos)

                fig_coord, axs_coord = plt.subplots(2, 2, figsize=(7, 7))
                pares_plot = [
                    (axs_coord[0, 0], 'D', label_prox_1, label_dist_1, f'{label_prox_1}(°)', f'{label_dist_1}(°)'),
                    (axs_coord[0, 1], 'E', label_prox_1, label_dist_1, f'{label_prox_1}(°)', f'{label_dist_1}(°)'),
                    (axs_coord[1, 0], 'D', label_prox_2, label_dist_2, f'{label_prox_2}(°)', f'{label_dist_2}(°)'),
                    (axs_coord[1, 1], 'E', label_prox_2, label_dist_2, f'{label_prox_2}(°)', f'{label_dist_2}(°)')
                ]

                for ax, lado, prox, dist, label_x, label_y in pares_plot:
                    ciclos_prox, ciclos_dist = np.array(dados_grp[f"{prox}_{lado}"]), np.array(dados_grp[f"{dist}_{lado}"])
                    if len(ciclos_prox) > 0 and len(ciclos_dist) > 0:
                        media_prox, media_dist = np.mean(ciclos_prox, axis=0), np.mean(ciclos_dist, axis=0)
                        ax.plot(media_prox, media_dist, color='blue' if lado=='E' else 'red', lw=2)
                        ax.scatter(media_prox[0], media_dist[0], color='green', s=60, zorder=5)
                        ax.scatter(media_prox[60], media_dist[60], color='orange', marker='X', s=60, zorder=5)
                    
                    ax.set_xlabel(label_x, fontsize=9)
                    ax.set_ylabel(label_y, fontsize=9)
                    ax.set_title(f"{prox}-{dist} ({lado})", fontweight='bold', fontsize=10)
                    ax.grid(True, linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig_coord)
                plt.close(fig_coord)

        st.markdown("---")
        st.markdown("### 2. Distribuição por Fases (Apoio vs. Balanço)")
        sub_tab_apoio, sub_tab_balanco = st.tabs(["🦵 Fase de Apoio (0-60%)", "✈️ Fase de Balanço (60-100%)"])
        
        def plot_fase_especifica(container, inicio, fim, titulo_fase):
            cols_fase = container.columns(len(grupos_estudo))
            for idx, grp in enumerate(grupos_estudo):
                with cols_fase[idx]:
                    st.markdown(f"<h6 style='text-align:center;'>{titulo_fase}: {grp}</h6>", unsafe_allow_html=True)
                    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
                    labels_pares = list(freq_acumulada[grp].keys())
                    m_prox, m_fase, m_dist, m_anti = [], [], [], []

                    for par in labels_pares:
                        chave_par_interna = next(old for old, new in pares_map if new == par)
                        f_prox, f_fase, f_dist, f_anti = 0, 0, 0, 0
                        contagem = 0
                        
                        for p in [proc for proc in st.session_state.processadores if proc.grupo == grp]:
                            try:
                                fatia = p.coord_vetorial_series.get(chave_par_interna, [])[inicio:fim]
                                total_fatia = len(fatia)
                                if total_fatia > 0:
                                    f_prox += fatia.count('Proximal') / total_fatia
                                    f_fase += fatia.count('EmFase') / total_fatia
                                    f_dist += fatia.count('Distal') / total_fatia
                                    f_anti += fatia.count('AntiFase') / total_fatia
                                    contagem += 1
                            except Exception: 
                                continue
                        
                        denom = contagem if contagem > 0 else 1
                        m_prox.append((f_prox/denom)*100)
                        m_fase.append((f_fase/denom)*100)
                        m_dist.append((f_dist/denom)*100)
                        m_anti.append((f_anti/denom)*100)

                    x, width = np.arange(len(labels_pares)), 0.55
                    bar1 = ax_bar.bar(x, m_prox, width, label='Proximal', color=padroes['Proximal'])
                    bar2 = ax_bar.bar(x, m_fase, width, bottom=m_prox, label='Em Fase', color=padroes['EmFase'])
                    bar3 = ax_bar.bar(x, m_dist, width, bottom=np.add(m_prox, m_fase), label='Distal', color=padroes['Distal'])
                    bar4 = ax_bar.bar(x, m_anti, width, bottom=np.add(np.add(m_prox, m_fase), m_dist), label='Anti-Fase', color=padroes['AntiFase'])

                    for bar_group in [bar1, bar2, bar3, bar4]:
                        lbls = [f"{v.get_height():.1f}%" if v.get_height() > 2.0 else "" for v in bar_group]
                        ax_bar.bar_label(bar_group, labels=lbls, label_type='center', color='black', fontweight='bold', fontsize=9)

                    ax_bar.set_ylabel('Frequência (%)', fontweight='bold')
                    ax_bar.set_xticks(x)
                    ax_bar.set_xticklabels(labels_pares, fontweight='bold', rotation=15, fontsize=8)
                    ax_bar.set_ylim(0, 105)
                    ax_bar.spines['top'].set_visible(False)
                    ax_bar.spines['right'].set_visible(False)
                    
                    if idx == len(grupos_estudo) - 1:
                        leg = ax_bar.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title="Padrões")
                        leg.get_title().set_fontweight('bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig_bar)
                    plt.close(fig_bar)

        plot_fase_especifica(sub_tab_apoio, 0, 60, "Apoio")
        plot_fase_especifica(sub_tab_balanco, 60, 100, "Balanço")

        st.markdown("---")
        st.markdown("### 3. Índices de Estabilidade e Fluidez Motora")
        col_cav, col_trans = st.columns(2)
        
        def plot_grouped_bars(ax, dict_data, titulo, ylabel):
            labels_grupos = list(dict_data.keys())
            means = [np.mean(dict_data[g]) if dict_data[g] else 0 for g in labels_grupos]
            stds = [np.std(dict_data[g]) if dict_data[g] else 0 for g in labels_grupos]
            cores = [cor_g1, cor_g2] if len(labels_grupos) > 1 else [cor_g1]
            bars = ax.bar(np.arange(len(labels_grupos)), means, yerr=stds, capsize=8, color=cores, edgecolor='black', alpha=0.9, width=0.6)
            ax.bar_label(bars, fmt='%.1f', padding=3, fontweight='bold')
            ax.set_xticks(np.arange(len(labels_grupos)))
            ax.set_xticklabels(labels_grupos, fontweight='bold')
            ax.set_title(titulo, fontweight='bold', fontsize=12)
            ax.set_ylabel(ylabel)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        with col_cav:
            fig_cav, axs_cav = plt.subplots(1, 2, figsize=(10, 5))
            plot_grouped_bars(axs_cav[0], cav_data[label_cav_1], f"Variabilidade (CAV)\n{label_prox_1}-{label_dist_1}", "Graus (°)")
            plot_grouped_bars(axs_cav[1], cav_data[label_cav_2], f"Variabilidade (CAV)\n{label_prox_2}-{label_dist_2}", "Graus (°)")
            plt.tight_layout()
            st.pyplot(fig_cav)
            plt.close(fig_cav)
            
        with col_trans:
            fig_tr, axs_tr = plt.subplots(1, 2, figsize=(10, 5))
            plot_grouped_bars(axs_tr[0], trans_data[label_cav_1], f"Taxa de Transições\n{label_prox_1}-{label_dist_1}", "Mudanças / Ciclo")
            plot_grouped_bars(axs_tr[1], trans_data[label_cav_2], f"Taxa de Transições\n{label_prox_2}-{label_dist_2}", "Mudanças / Ciclo")
            plt.tight_layout()
            st.pyplot(fig_tr)
            plt.close(fig_tr)

    with tab4:
        st.subheader("Gerador de GIFs e Biofeedback Visual")
        st.info("Escolha os arquivos que deseja animar e a velocidade de reprodução.")
        lista_nomes = [p.nome_arq for p in st.session_state.processadores]
        col_selecao, col_vel = st.columns([2, 1])
        with col_selecao: 
            selecionados = st.multiselect("Selecione os arquivos para animar:", lista_nomes)
        with col_vel: 
            vel_opcao = st.radio("Velocidade de Reprodução:", ["100% (Normal)", "75% (Lenta)", "50% (Muito Lenta)"])
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
                            with open(tmp_gif_path, "rb") as file_gif: 
                                st.download_button(f"📥 Baixar GIF", data=file_gif, file_name=f"{p.nome_arq.split('.')[0]}_3D.gif", mime="image/gif")
                        else: 
                            st.error(f"Falha: {msg}")

    with tab5:
        st.subheader("Análise Estatística Avançada (Comparação)")
        grupos_estudo = sorted(list(set([p.grupo for p in st.session_state.processadores])))
        cor_g1, cor_g2 = '#a8c8f9', '#f9a8a8'
        cores_grupos = {grupos_estudo[0]: cor_g1}
        if len(grupos_estudo) > 1: 
            cores_grupos[grupos_estudo[1]] = cor_g2

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

        st.markdown("### 1. Parâmetros Espaço-Temporais")

        def gerar_colunas_passo_norm(dict_dados, titulo):
            fig, ax = plt.subplots(figsize=(8, 6))
            labels = list(dict_dados.keys())
            means = [np.nanmean(dict_dados[l]) if dict_dados[l] else 0 for l in labels]
            stds = [np.nanstd(dict_dados[l]) if dict_dados[l] else 0 for l in labels]
            
            cores = [cor_g1 if grupos_estudo[0] in l else cor_g2 for l in labels]
            bars = ax.bar(labels, means, yerr=stds, capsize=10, color=cores, edgecolor='black', alpha=0.8)
            
            for i, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 1, f"Média: {means[i]:.0f}%\nDP: ±{stds[i]:.1f}%", ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

            ax.set_title(titulo, fontweight='bold', fontsize=14)
            ax.set_ylabel("Porcentagem da Estatura (%)", fontsize=12)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            return fig
        
        def gerar_boxplot_isolado(dict_dados, titulo, ylabel):
            fig, ax = plt.subplots(figsize=(8, 6)) 
            labels = list(dict_dados.keys())
            dados_limpos = [dict_dados[l] for l in labels if len(dict_dados[l]) > 0]
            labels_limpos = [l for l in labels if len(dict_dados[l]) > 0]
            
            if dados_limpos:
                bp = ax.boxplot(dados_limpos, patch_artist=True, labels=labels_limpos)
                for i, patch in enumerate(bp['boxes']): 
                    grp_name = labels_limpos[i].split('(')[0] if '(' in labels_limpos[i] else labels_limpos[i]
                    patch.set_facecolor(cores_grupos.get(grp_name, '#dddddd'))
                for median in bp['medians']: 
                    median.set(color='black', linewidth=2)
                for i, d in enumerate(dados_limpos):
                    media, dp, mediana = np.mean(d), np.std(d), np.median(d)
                    ax.text(i + 1.10, mediana, f"M: {media:.1f}\nDP: {dp:.1f}", ha='left', va='center', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'))
                
                ax.set_xlim(0.5, len(dados_limpos) + 0.9)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin - (ymax-ymin)*0.1, ymax + (ymax-ymin)*0.1)
                ax.set_title(titulo, fontweight='bold', fontsize=14)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            return fig

        dict_vel = {g: v_vel[g] for g in grupos_estudo}
        dict_ap = {}; dict_fc = {}; dict_ps = {}; dict_ps_norm = {}
        for g in grupos_estudo:
            dict_ap[f"{g}(D)"] = v_ap_d[g]; dict_ap[f"{g}(E)"] = v_ap_e[g]
            dict_fc[f"{g}(D)"] = v_fc_d[g]; dict_fc[f"{g}(E)"] = v_fc_e[g]
            dict_ps[f"{g}(D)"] = v_ps_d[g]; dict_ps[f"{g}(E)"] = v_ps_e[g]
            dict_ps_norm[f"{g}(D)"] = v_ps_norm_d[g]; dict_ps_norm[f"{g}(E)"] = v_ps_norm_e[g]
            
        col_box1, col_box2 = st.columns(2)
        with col_box1:
            fig1 = gerar_boxplot_isolado(dict_vel, "Velocidade Média", "m/s")
            st.pyplot(fig1)
            plt.close(fig1)
            fig2 = gerar_boxplot_isolado(dict_fc, "Foot Clearance", "mm")
            st.pyplot(fig2)
            plt.close(fig2)
        with col_box2:
            fig3 = gerar_boxplot_isolado(dict_ap, "Fase de Apoio", "% do Ciclo")
            st.pyplot(fig3)
            plt.close(fig3)
            fig4 = gerar_boxplot_isolado(dict_ps, "Comprimento do Passo (Absoluto)", "mm")
            st.pyplot(fig4)
            plt.close(fig4)
            
        st.markdown("---")
        col_box3, col_box4 = st.columns(2)
        with col_box3:
            fig5 = gerar_colunas_passo_norm(dict_ps_norm, "Comprimento do Passo (% Altura)")
            st.pyplot(fig5)
            plt.close(fig5)

    
