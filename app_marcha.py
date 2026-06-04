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
import warnings

# Ignora avisos de médias com slices vazios (comum ao inserir NaNs propositais)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA E ESTILO MATPLOTLIB (PAPER STYLE)
# =============================================================================
st.set_page_config(page_title="GPBIO - Biomecânica Clínica", layout="wide", page_icon="🚶")

plt.rcParams.update({
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.grid': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'font.family': 'sans-serif',
    'font.size': 10
})

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
            self.tempo_ciclo_s = self._calcular_tempo_medio_ciclo()

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
                        if not np.isnan(val_d) and val_d > 0: self.passo_norm['D'] = ((val_d / 1000.0) / altura_m) * 100.0
                        if not np.isnan(val_e) and val_e > 0: self.passo_norm['E'] = ((val_e / 1000.0) / altura_m) * 100.0
            
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
                if np.isnan(sinal_d).all(): continue
                s_temp = pd.Series(sinal_d).interpolate(limit_direction='both').bfill().ffill()
                try: out[ax, m, :] = signal.filtfilt(b, a, s_temp.to_numpy())
                except Exception: out[ax, m, :] = s_temp
        return out

    def _ang_sagital_vert(self, p_prox, p_dist):
        if p_prox is None or p_dist is None: return np.nan
        dx = (p_dist[0] - p_prox[0]) * self.dir_x
        dz = p_dist[2] - p_prox[2] 
        return np.degrees(np.arctan2(dx, -dz))

    def _calcular_angulos_segmentares(self):
        res = {k: [] for k in ['Coxa_D','Perna_D','Pe_D','Coxa_E','Perna_E','Pe_E']}
        for f in range(self.n_frames):
            for lado, l in [('D', 'R'), ('E', 'L')]:
                h = self._get(f'{l}IAS',f); k = self._mid(f'{l}LE', f'{l}ME',f); a = self._mid(f'{l}ML', f'{l}MM',f)
                p = self._get(f'{l}FT1', f); cal = self._get(f'{l}CAL',f)
                
                res[f'Coxa_{lado}'].append(self._ang_sagital_vert(h, k))
                res[f'Perna_{lado}'].append(self._ang_sagital_vert(k, a))
                res[f'Pe_{lado}'].append(self._ang_sagital_vert(cal, p) - 90)
        return pd.DataFrame(res)

    def _calcular_angulos(self):
        res = {k: [] for k in ['Quad_D','Joel_D','Torn_D','Quad_E','Joel_E','Torn_E']}
        for f in range(self.n_frames):
            for lado in ['D', 'E']:
                coxa = self.segmentos_df[f'Coxa_{lado}'][f]; perna = self.segmentos_df[f'Perna_{lado}'][f]; pe = self.segmentos_df[f'Pe_{lado}'][f]
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
            if ciclos_apoio: res[lado]['Apoio'] = np.mean(ciclos_apoio); res[lado]['Balanco'] = 100.0 - np.mean(ciclos_apoio)
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

    def extrair_ciclos_normalizados(self, vetor_dados, eventos_hs, eventos_to=None, pontos=101):
        ciclos = []
        if len(eventos_hs) < 2: return []
        
        pontos = int(pontos)
        idx_to_norm = int(round((pontos - 1) * 0.60))
        pts_apoio = int(idx_to_norm + 1)
        pts_balanco = int(pontos - idx_to_norm)
        
        for i in range(len(eventos_hs) - 1):
            hs_atual = eventos_hs[i]
            hs_prox = eventos_hs[i+1]
            if hs_prox >= len(vetor_dados): continue
            
            to_valido = None
            if eventos_to is not None:
                tos_no_ciclo = [t for t in eventos_to if hs_atual < t < hs_prox]
                if tos_no_ciclo:
                    to_valido = tos_no_ciclo[0]
            
            if to_valido is not None and hs_atual < to_valido < hs_prox:
                fase_apoio = vetor_dados[hs_atual:to_valido+1] 
                fase_apoio_norm = np.interp(np.linspace(0, len(fase_apoio)-1, pts_apoio), np.arange(len(fase_apoio)), fase_apoio)
                
                fase_balanco = vetor_dados[to_valido:hs_prox+1]
                fase_balanco_norm = np.interp(np.linspace(0, len(fase_balanco)-1, pts_balanco), np.arange(len(fase_balanco)), fase_balanco)
                
                ciclo_norm = np.concatenate((fase_apoio_norm, fase_balanco_norm[1:]))
                ciclos.append(ciclo_norm)
            else:
                ciclo_bruto = vetor_dados[hs_atual:hs_prox+1]
                if len(ciclo_bruto) < 2: continue
                ciclos.append(np.interp(np.linspace(0, len(ciclo_bruto)-1, pontos), np.arange(len(ciclo_bruto)), ciclo_bruto))
        return ciclos

    def _calcular_coordenacao_vetorial(self):
        res = {}; 
        self.coord_vetorial_series = {} 
        self.coord_vetorial_cav = {} ### NOVO: Dicionário nativo para armazenar o CAV contínuo
        
        for lado in ['D', 'E']:
            hss = self.eventos[lado]['HS']
            tos = self.eventos[lado]['TO']
            if len(hss) < 2: continue
            
            pares = [
                (f'Quad_Joel_{lado}', f'Quad_{lado}', f'Joel_{lado}', self.angulos_df),
                (f'Joel_Torn_{lado}', f'Joel_{lado}', f'Torn_{lado}', self.angulos_df),
                (f'Coxa_Perna_{lado}', f'Coxa_{lado}', f'Perna_{lado}', self.segmentos_df),
                (f'Perna_Pe_{lado}', f'Perna_{lado}', f'Pe_{lado}', self.segmentos_df)
            ]
            
            for nome_par, col_prox, col_dist, df_ref in pares:
                prox_norm_list = self.extrair_ciclos_normalizados(df_ref[col_prox].values, hss, tos)
                dist_norm_list = self.extrair_ciclos_normalizados(df_ref[col_dist].values, hss, tos)

                if not prox_norm_list or not dist_norm_list: continue
                
                res[nome_par] = {'Proximal': np.nan, 'Distal': np.nan, 'EmFase': np.nan, 'AntiFase': np.nan}
                freqs = {'Proximal': [], 'Distal': [], 'EmFase': [], 'AntiFase': []}
                cas = []
                
                for prox_norm, dist_norm in zip(prox_norm_list, dist_norm_list):
                    dx, dy = -np.diff(prox_norm), -np.diff(dist_norm)
                    ca_norm = np.mod(np.degrees(np.arctan2(dy, dx)), 360)
                    ca_norm = np.append(ca_norm, ca_norm[-1] if not np.isnan(ca_norm[-1]) else np.nan) 
                    cas.append(ca_norm)
                    
                    counts = {'Proximal': 0, 'Distal': 0, 'EmFase': 0, 'AntiFase': 0}
                    valid_frames = 0
                    for a in ca_norm:
                        if np.isnan(a): continue
                        valid_frames += 1
                        if (0 <= a < 22.5) or (337.5 <= a <= 360) or (157.5 <= a < 202.5): counts['Proximal'] += 1
                        elif (22.5 <= a < 67.5) or (202.5 <= a < 247.5): counts['EmFase'] += 1
                        elif (67.5 <= a < 112.5) or (247.5 <= a < 292.5): counts['Distal'] += 1
                        else: counts['AntiFase'] += 1
                        
                    if valid_frames > 0:
                        for k in counts: freqs[k].append((counts[k] / valid_frames) * 100)
                
                if not cas: continue
                
                for k in freqs: 
                    res[nome_par][k] = np.mean(freqs[k]) if len(freqs[k]) > 0 else np.nan
                
                rad_cas = np.radians(np.array(cas))
                s_m = np.nanmean(np.sin(rad_cas), axis=0)
                c_m = np.nanmean(np.cos(rad_cas), axis=0)
                ang_m = np.mod(np.degrees(np.arctan2(s_m, c_m)), 360)
                
                ### NOVO: Aproveitamos o s_m e c_m que já foram processados! Cálculo muito mais rápido.
                R = np.clip(np.sqrt(s_m**2 + c_m**2), 1e-10, 1.0) 
                self.coord_vetorial_cav[nome_par] = np.degrees(np.sqrt(-2 * np.log(R)))
                ### FIM DO NOVO
                
                fatia_media = []
                for a in ang_m:
                    if np.isnan(a): fatia_media.append('Ruido')
                    elif (0 <= a < 22.5) or (337.5 <= a <= 360) or (157.5 <= a < 202.5): fatia_media.append('Proximal')
                    elif (22.5 <= a < 67.5) or (202.5 <= a < 247.5): fatia_media.append('EmFase')
                    elif (67.5 <= a < 112.5) or (247.5 <= a < 292.5): fatia_media.append('Distal')
                    else: fatia_media.append('AntiFase')
                self.coord_vetorial_series[nome_par] = fatia_media
        return res

# =============================================================================
# MÓDULO VISUAL MODULAR (GIFs Estilo Qualisys QTM) 
# =============================================================================
class GeradorVisual:
    def __init__(self, processador, nome_original, opt_bussolas=False, opt_ang=False, opt_aa=False, opt_ca=False):
        self.proc = processador
        self.nome_arq = nome_original
        self.box = {'x': (-1000, 1000), 'y': (-1000, 1000), 'z': (0, 2000)}
        self.opt_bussolas = opt_bussolas
        self.opt_ang = opt_ang
        self.opt_aa = opt_aa
        self.opt_ca = opt_ca

    def _get_f(self, n, f):
        try: return self.proc._get(n, f)
        except Exception: return None
    
    def _mid_f(self, n1, n2, f):
        p1 = self._get_f(n1, f); p2 = self._get_f(n2, f)
        if p1 is not None and p2 is not None: return (p1 + p2) / 2.0
        return p1 if p1 is not None else p2

    def _get_primeiro_valido(self, lista_nomes, f):
        for nome in lista_nomes:
            val = self._get_f(nome, f)
            if val is not None: return val
        return None

    def montar_frame(self, f):
        s = {}
        
        # =========================================================
        # 1. PELVE COMPLETA (AZUL) - Todas as conexões
        # =========================================================
        rias = self._get_primeiro_valido(['RIAS', 'RASI'], f)
        lias = self._get_primeiro_valido(['LIAS', 'LASI'], f)
        rips = self._get_primeiro_valido(['RIPS', 'RPSI'], f)
        lips = self._get_primeiro_valido(['LIPS', 'LPSI'], f)
        rict = self._get_primeiro_valido(['RICT'], f)
        lict = self._get_primeiro_valido(['LICT'], f)
        sacr = self._get_primeiro_valido(['SACR', 'SACRUM'], f)
        
        if rias is not None and lias is not None: s['Pelve_Frente'] = [rias, lias]
        if rips is not None and lips is not None: s['Pelve_Tras'] = [rips, lips]
        
        if rict is not None:
            if rias is not None: s['Pelve_Crista_Ant_D'] = [rict, rias]
            if rips is not None: s['Pelve_Crista_Post_D'] = [rict, rips]
        elif rias is not None and rips is not None:
            s['Pelve_Lat_D'] = [rias, rips]
            
        if lict is not None:
            if lias is not None: s['Pelve_Crista_Ant_E'] = [lict, lias]
            if lips is not None: s['Pelve_Crista_Post_E'] = [lict, lips]
        elif lias is not None and lips is not None:
            s['Pelve_Lat_E'] = [lias, lips]

        if sacr is not None:
            if rips is not None: s['Pelve_Sacro_D'] = [rips, sacr]
            if lips is not None: s['Pelve_Sacro_E'] = [lips, sacr]
            if rips is None and rias is not None: s['Pelve_Sacro_Alt_D'] = [rias, sacr]
            if lips is None and lias is not None: s['Pelve_Sacro_Alt_E'] = [lias, sacr]
        
        quad_d = rias if rias is not None else (rips if rips is not None else (rict if rict is not None else sacr))
        quad_e = lias if lias is not None else (lips if lips is not None else (lict if lict is not None else sacr))

        # =========================================================
        # 2. JOELHOS 
        # =========================================================
        kd = self._mid_f('RLE', 'RME', f)
        if kd is None: kd = self._get_primeiro_valido(['RLE', 'RME', 'RKN'], f)
        
        ke = self._mid_f('LLE', 'LME', f)
        if ke is None: ke = self._get_primeiro_valido(['LLE', 'LME', 'LKN'], f)

        # =========================================================
        # 3. COXAS DIRETAS
        # =========================================================
        if quad_d is not None and kd is not None: s['Coxa_D'] = [quad_d, kd]
        if quad_e is not None and ke is not None: s['Coxa_E'] = [quad_e, ke]

        # =========================================================
        # 4. TORNOZELOS E PERNAS
        # =========================================================
        td = self._mid_f('RML', 'RMM', f)
        if td is None: td = self._get_primeiro_valido(['RML', 'RMM', 'RANK'], f)
        
        te = self._mid_f('LML', 'LMM', f)
        if te is None: te = self._get_primeiro_valido(['LML', 'LMM', 'LANK'], f)

        if kd is not None and td is not None: s['Perna_D'] = [kd, td]
        if ke is not None and te is not None: s['Perna_E'] = [ke, te]

        # =========================================================
        # 5. PÉS EM PIRÂMIDE (Base triangular fechada)
        # =========================================================
        rcal = self._get_primeiro_valido(['RCAL', 'RHEE'], f)
        lcal = self._get_primeiro_valido(['LCAL', 'LHEE'], f)
        rft1 = self._get_primeiro_valido(['RFT1', 'RTOE'], f)
        lft1 = self._get_primeiro_valido(['LFT1', 'LTOE'], f)
        rft5 = self._get_primeiro_valido(['RFT5'], f)
        lft5 = self._get_primeiro_valido(['LFT5'], f)

        if td is not None and rcal is not None: s['Pe_Tornoz_Calc_D'] = [td, rcal]
        if td is not None and rft1 is not None: s['Pe_Tornoz_M1_D'] = [td, rft1]
        if td is not None and rft5 is not None: s['Pe_Tornoz_M5_D'] = [td, rft5]
        if rcal is not None and rft1 is not None: s['Pe_Sola_Medial_D'] = [rcal, rft1]
        if rcal is not None and rft5 is not None: s['Pe_Sola_Lateral_D'] = [rcal, rft5]
        if rft1 is not None and rft5 is not None: s['Pe_Sola_Frente_D'] = [rft1, rft5]

        if te is not None and lcal is not None: s['Pe_Tornoz_Calc_E'] = [te, lcal]
        if te is not None and lft1 is not None: s['Pe_Tornoz_M1_E'] = [te, lft1]
        if te is not None and lft5 is not None: s['Pe_Tornoz_M5_E'] = [te, lft5]
        if lcal is not None and lft1 is not None: s['Pe_Sola_Medial_E'] = [lcal, lft1]
        if lcal is not None and lft5 is not None: s['Pe_Sola_Lateral_E'] = [lcal, lft5]
        if lft1 is not None and lft5 is not None: s['Pe_Sola_Frente_E'] = [lft1, lft5]
        
        return s

    def _criar_bussola(self, ax, titulo):
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.4, 1.4); ax.axis('off'); ax.set_aspect('equal')
        ax.text(0, 1.35, titulo, ha='center', va='center', fontsize=9, fontweight='bold')
        txt = ax.text(0, -1.35, "-", ha='center', va='center', fontsize=10, fontweight='bold')
        categorias = [((0, 22.5), '#e74c3c'), ((337.5, 360), '#e74c3c'), ((157.5, 202.5), '#e74c3c'),
                      ((22.5, 67.5), '#2ecc71'), ((202.5, 247.5), '#2ecc71'),
                      ((67.5, 112.5), '#3498db'), ((247.5, 292.5), '#3498db'),
                      ((112.5, 157.5), '#f1c40f'), ((292.5, 337.5), '#f1c40f')]
        for (t1, t2), cor in categorias: 
            ax.add_patch(mpatches.Wedge((0,0), 1.0, t1, t2, facecolor=cor, alpha=0.35, edgecolor='white', lw=1))
        ax.plot([0], [0], marker='o', color='black', markersize=4)
        ptr, = ax.plot([], [], color='black', lw=2.5)
        return ptr, txt

    def salvar(self, caminho_final, step=3, fps_anim=20):
        try:
            if self.proc.segmentos_df is None or len(self.proc.segmentos_df) < 5:
                return False, "Dados insuficientes no dataframe (menos de 5 frames)."
            
            total_frames = len(self.proc.segmentos_df) - 1
            
            cx_d = self.proc.segmentos_df['Coxa_D'].values[:total_frames]
            pn_d = self.proc.segmentos_df['Perna_D'].values[:total_frames]
            pe_d = self.proc.segmentos_df['Pe_D'].values[:total_frames]
            cx_e = self.proc.segmentos_df['Coxa_E'].values[:total_frames]
            pn_e = self.proc.segmentos_df['Perna_E'].values[:total_frames]
            pe_e = self.proc.segmentos_df['Pe_E'].values[:total_frames]

            def calc_ca(prox, dist):
                dx, dy = -np.diff(prox), -np.diff(dist)
                ca = np.mod(np.degrees(np.arctan2(dy, dx)), 360)
                ca = np.append(ca, ca[-1] if len(ca) > 0 else 0)
                return ca

            ca_cp_d = calc_ca(cx_d, pn_d)
            ca_pp_d = calc_ca(pn_d, pe_d)
            ca_cp_e = calc_ca(cx_e, pn_e)
            ca_pp_e = calc_ca(pn_e, pe_e)
            
            largura = 7
            ratios = []
            if self.opt_bussolas: ratios.append(1); largura += 2
            ratios.append(2) 
            opts_dir = sum([self.opt_ang, self.opt_aa, self.opt_ca])
            if opts_dir > 0: ratios.append(1.5); largura += 4
            
            fig = plt.figure(figsize=(largura, 6))
            gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios)
            col = 0
            
            bussolas = {}
            if self.opt_bussolas:
                gs_l = gs[0, col].subgridspec(2, 2)
                bussolas['cp_d'] = self._criar_bussola(fig.add_subplot(gs_l[0,0]), "Coxa-Perna(D)")
                bussolas['cp_e'] = self._criar_bussola(fig.add_subplot(gs_l[0,1]), "Coxa-Perna(E)")
                bussolas['pp_d'] = self._criar_bussola(fig.add_subplot(gs_l[1,0]), "Perna-Pé(D)")
                bussolas['pp_e'] = self._criar_bussola(fig.add_subplot(gs_l[1,1]), "Perna-Pé(E)")
                col += 1

            ax_3d = fig.add_subplot(gs[0, col], projection='3d')
            ax_3d.set_xlim(self.box['x']); ax_3d.set_ylim(self.box['y']); ax_3d.set_zlim(self.box['z'])
            ax_3d.view_init(elev=20, azim=135)
            ax_3d.set_title(self.nome_arq)
            
            # Mantém a grade de fundo mas remove as legendas numeradas dos eixos
            ax_3d.grid(True)
            ax_3d.set_xticklabels([])
            ax_3d.set_yticklabels([])
            ax_3d.set_zticklabels([])
            
            col += 1

            dots = {}
            if opts_dir > 0:
                gs_r = gs[0, col].subgridspec(opts_dir, 1, hspace=0.4)
                r = 0
                x_axis = np.arange(total_frames)
                
                if self.opt_ang:
                    gs_ang = gs_r[r, 0].subgridspec(1, 3)
                    ax1 = fig.add_subplot(gs_ang[0,0]); ax1.plot(x_axis, cx_d, 'k-', alpha=0.4); ax1.set_title("Coxa(°)", fontsize=9); ax1.set_xticks([])
                    ax2 = fig.add_subplot(gs_ang[0,1]); ax2.plot(x_axis, pn_d, 'k-', alpha=0.4); ax2.set_title("Perna(°)", fontsize=9); ax2.set_xticks([])
                    ax3 = fig.add_subplot(gs_ang[0,2]); ax3.plot(x_axis, pe_d, 'k-', alpha=0.4); ax3.set_title("Pé(°)", fontsize=9); ax3.set_xticks([])
                    dots['ang_c'], = ax1.plot([], [], 'ro')
                    dots['ang_p'], = ax2.plot([], [], 'ro')
                    dots['ang_f'], = ax3.plot([], [], 'ro')
                    r += 1
                    
                if self.opt_aa:
                    gs_aa = gs_r[r, 0].subgridspec(1, 2)
                    ax1 = fig.add_subplot(gs_aa[0,0]); ax1.plot(cx_d, pn_d, 'k-', alpha=0.4); ax1.set_title("AA Coxa-Perna", fontsize=9)
                    ax2 = fig.add_subplot(gs_aa[0,1]); ax2.plot(pn_d, pe_d, 'k-', alpha=0.4); ax2.set_title("AA Perna-Pé", fontsize=9)
                    dots['aa_cp'], = ax1.plot([], [], 'ro')
                    dots['aa_pp'], = ax2.plot([], [], 'ro')
                    r += 1
                    
                if self.opt_ca:
                    gs_ca = gs_r[r, 0].subgridspec(1, 2)
                    ax1 = fig.add_subplot(gs_ca[0,0]); ax1.plot(x_axis, ca_cp_d, 'k-', alpha=0.4); ax1.set_title("CA Coxa-Perna", fontsize=9); ax1.set_ylim(0,360); ax1.set_yticks([0,180,360]); ax1.set_xticks([])
                    ax2 = fig.add_subplot(gs_ca[0,1]); ax2.plot(x_axis, ca_pp_d, 'k-', alpha=0.4); ax2.set_title("CA Perna-Pé", fontsize=9); ax2.set_ylim(0,360); ax2.set_yticks([0,180,360]); ax2.set_xticks([])
                    dots['ca_cp'], = ax1.plot([], [], 'ro')
                    dots['ca_pp'], = ax2.plot([], [], 'ro')

            linhas_3d = {}

            def set_bussola(chave, angulo):
                if chave in bussolas:
                    ptr, txt = bussolas[chave]
                    if np.isnan(angulo):
                        ptr.set_data([], []); txt.set_text("-"); txt.set_color("gray")
                    else:
                        ptr.set_data([0, np.cos(np.radians(angulo))], [0, np.sin(np.radians(angulo))])
                        a = angulo % 360
                        if (0 <= a < 22.5) or (337.5 <= a <= 360) or (157.5 <= a < 202.5): lbl, c = "PROXIMAL", '#e74c3c'
                        elif (22.5 <= a < 67.5) or (202.5 <= a < 247.5): lbl, c = "EM FASE", '#2ecc71'
                        elif (67.5 <= a < 112.5) or (247.5 <= a < 292.5): lbl, c = "DISTAL", '#3498db'
                        else: lbl, c = "ANTI-FASE", '#f1c40f'
                        txt.set_text(lbl); txt.set_color(c)

            def update(i):
                seg = self.montar_frame(i)
                for k in list(linhas_3d):
                    if k not in seg: linhas_3d[k].remove(); del linhas_3d[k]
                
                for n, (p1, p2) in seg.items():
                    # NOVO ESQUEMA INVERTIDO: Direito = Roxo, Esquerdo = Verde
                    if 'Pelve' in n: cor_bolinha = '#3498db' # Azul
                    elif '_D' in n: cor_bolinha = '#9b59b6'  # Roxo
                    elif '_E' in n: cor_bolinha = '#2ecc71'  # Verde
                    else: cor_bolinha = 'gray'
                    
                    x_plot = [-p1[0], -p2[0]] # Inversão Horizontal
                    y_plot = [p1[1], p2[1]]
                    z_plot = [p1[2], p2[2]]
                    
                    if n in linhas_3d:
                        linhas_3d[n].set_data(x_plot, y_plot)
                        linhas_3d[n].set_3d_properties(z_plot)
                    else: 
                        # Ordem de Sobreposição: zorder=4 garante a pelve por cima das coxas.
                        z_ord = 4 if 'Pelve' in n else 3
                        
                        linhas_3d[n], = ax_3d.plot(x_plot, y_plot, z_plot, 
                                                   color='#f1c40f',             # Osso (Amarelo)
                                                   linewidth=3.0,               
                                                   marker='o',                  
                                                   markerfacecolor=cor_bolinha, # Cor Interna da bolinha
                                                   markeredgecolor='black',     # Contorno de laboratório
                                                   markeredgewidth=0.8,
                                                   markersize=6,                
                                                   zorder=z_ord)
                
                if self.opt_bussolas:
                    set_bussola('cp_d', ca_cp_d[i])
                    set_bussola('cp_e', ca_cp_e[i])
                    set_bussola('pp_d', ca_pp_d[i])
                    set_bussola('pp_e', ca_pp_e[i])
                    
                if self.opt_ang:
                    dots['ang_c'].set_data([i], [cx_d[i]])
                    dots['ang_p'].set_data([i], [pn_d[i]])
                    dots['ang_f'].set_data([i], [pe_d[i]])
                if self.opt_aa:
                    dots['aa_cp'].set_data([cx_d[i]], [pn_d[i]])
                    dots['aa_pp'].set_data([pn_d[i]], [pe_d[i]])
                if self.opt_ca:
                    dots['ca_cp'].set_data([i], [ca_cp_d[i]])
                    dots['ca_pp'].set_data([i], [ca_pp_d[i]])
                    
                return list(linhas_3d.values()) + list(dots.values())

            ani = animation.FuncAnimation(fig, update, frames=range(0, total_frames, step), interval=50, blit=False)
            ani.save(caminho_final, writer='pillow', fps=fps_anim)
            return True, caminho_final
        
        except Exception as e:
            import traceback
            return False, f"Falha no processador visual: {str(e)}\n\nRastreio de Erro:\n{traceback.format_exc()}"
        finally:
            plt.close(fig)
            plt.close('all')
# =============================================================================
# FUNÇÕES GLOBAIS STREAMLIT DE FILTRAGEM E CÁLCULO
# =============================================================================

def exibir_resumo_processamento(processadores, nome_g1, nome_g2):
    """Conta e exibe na interface quantos arquivos foram processados por grupo."""
    total = len(processadores)
    
    if total == 0:
        st.warning("⚠️ Nenhum arquivo válido foi processado. Verifique os dados.")
        return
    
    qtd_g1 = sum(1 for p in processadores if p.grupo == nome_g1)
    qtd_g2 = sum(1 for p in processadores if p.grupo == nome_g2)
    
    msg = f"✅ **{total}** arquivos processados e prontos para os cálculos: "
    msg += f"**{qtd_g1}** no grupo '{nome_g1}' e **{qtd_g2}** no grupo '{nome_g2}'."
    
    st.success(msg)
    
def calcular_ca_serie(prox_raw, dist_raw, p_obj, hss, tos=None):
    """ Calcula Vector Coding normalizado (Cálculo Angular Bruto) """
    prox_norm_list = p_obj.extrair_ciclos_normalizados(prox_raw, hss, tos)
    dist_norm_list = p_obj.extrair_ciclos_normalizados(dist_raw, hss, tos)
    
    cas = []
    for p_n, d_n in zip(prox_norm_list, dist_norm_list):
        dx, dy = -np.diff(p_n), -np.diff(d_n)
        ca_norm = np.mod(np.degrees(np.arctan2(dy, dx)), 360)
        ca_norm = np.append(ca_norm, ca_norm[-1] if not np.isnan(ca_norm[-1]) else np.nan)
        cas.append(ca_norm)
    return cas

def media_circular(ciclos):
    if not ciclos: return []
    rad = np.radians(np.array(ciclos))
    s_m = np.nanmean(np.sin(rad), axis=0)
    c_m = np.nanmean(np.cos(rad), axis=0)
    return np.mod(np.degrees(np.arctan2(s_m, c_m)), 360)


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
        st.sidebar.error("Erro nas colunas. Falta ID ou ALTURA.")
        df_antropo = None

st.sidebar.markdown("---")
st.sidebar.markdown("### Sobre o Sistema")
st.sidebar.info("GPBIO: Análise Biomecânica de Marcha.")
st.sidebar.markdown("**Desenvolvido por Arthur Lins**")
    
if 'processadores' not in st.session_state: st.session_state.processadores = []

st.subheader("📁 Importação de Dados e Separação de Grupos")
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
        st.warning("Faça o upload de arquivos dinâmicos.")
    else:
        st.session_state.processadores = []
        progress_bar = st.progress(0)
        
        for i, (file, nome_grupo) in enumerate(arquivos_para_processar):
            file.seek(0) 
            with tempfile.NamedTemporaryFile(delete=False, suffix='.c3d') as tmp_file:
                tmp_file.write(file.read()); tmp_file.flush(); os.fsync(tmp_file.fileno()); tmp_path = tmp_file.name
                
            proc = ProcessadorCinematico(tmp_path, file.name, grupo=nome_grupo, df_antropo=df_antropo)
            if proc.valido: st.session_state.processadores.append(proc)
            else: st.error(f"Erro no arquivo {file.name}: {proc.erro_msg}")
            
            try: os.remove(tmp_path)
            except Exception: pass 
            progress_bar.progress((i + 1) / len(arquivos_para_processar))
            
        exibir_resumo_processamento(st.session_state.processadores, nome_g1, nome_g2)

if st.session_state.processadores:
    grupos_estudo = sorted(list(set([p.grupo for p in st.session_state.processadores])))
    cores_comp = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#e377c2']
    
    def obter_estilo(grp, idx):
        g = grp.lower()
        if 'control' in g: return 'black', '-', 1.2
        if 'parkinson' in g: return 'grey', '--', 1.2
        return cores_comp[idx % len(cores_comp)], '--', 1.2

    dados_curvas = {grp: {'Art': {}, 'Seg': {}, 'CA_Art': {}, 'CA_Seg': {}, 'Tempos': []} for grp in grupos_estudo}
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
            if hasattr(p, 'tempo_ciclo_s') and not np.isnan(p.tempo_ciclo_s):
                dados_curvas[grp]['Tempos'].append(p.tempo_ciclo_s)
                
            for lado in ['D', 'E']:
                hss = p.eventos[lado]['HS']
                tos = p.eventos[lado]['TO']
                
                cics_quad = p.extrair_ciclos_normalizados(p.angulos_df[f"Quad_{lado}"].values, hss, tos)
                cics_joel = p.extrair_ciclos_normalizados(p.angulos_df[f"Joel_{lado}"].values, hss, tos)
                cics_torn = p.extrair_ciclos_normalizados(p.angulos_df[f"Torn_{lado}"].values, hss, tos)
                cics_coxa = p.extrair_ciclos_normalizados(p.segmentos_df[f"Coxa_{lado}"].values, hss, tos)
                cics_perna = p.extrair_ciclos_normalizados(p.segmentos_df[f"Perna_{lado}"].values, hss, tos)
                cics_pe = p.extrair_ciclos_normalizados(p.segmentos_df[f"Pe_{lado}"].values, hss, tos)
                
                dados_curvas[grp]['Art'][f"Quad_{lado}"].extend(cics_quad)
                dados_curvas[grp]['Art'][f"Joel_{lado}"].extend(cics_joel)
                dados_curvas[grp]['Art'][f"Torn_{lado}"].extend(cics_torn)
                dados_curvas[grp]['Seg'][f"Coxa_{lado}"].extend(cics_coxa)
                dados_curvas[grp]['Seg'][f"Perna_{lado}"].extend(cics_perna)
                dados_curvas[grp]['Seg'][f"Pe_{lado}"].extend(cics_pe)
                
                dados_curvas[grp]['CA_Art'][f"Quad_Joel_{lado}"].extend(calcular_ca_serie(p.angulos_df[f"Quad_{lado}"].values, p.angulos_df[f"Joel_{lado}"].values, p, hss, tos))
                dados_curvas[grp]['CA_Art'][f"Joel_Torn_{lado}"].extend(calcular_ca_serie(p.angulos_df[f"Joel_{lado}"].values, p.angulos_df[f"Torn_{lado}"].values, p, hss, tos))
                dados_curvas[grp]['CA_Seg'][f"Coxa_Perna_{lado}"].extend(calcular_ca_serie(p.segmentos_df[f"Coxa_{lado}"].values, p.segmentos_df[f"Perna_{lado}"].values, p, hss, tos))
                dados_curvas[grp]['CA_Seg'][f"Perna_Pe_{lado}"].extend(calcular_ca_serie(p.segmentos_df[f"Perna_{lado}"].values, p.segmentos_df[f"Pe_{lado}"].values, p, hss, tos))

    tab_cin, tab_aa, tab_ca, tab_paper, tab_freq, tab_est, tab_anim, tab_tab = st.tabs([
        "📈 Cinemática", 
        "🔄 Angle-Angle", 
        "📈 Coupling Angle", 
        "📄 Plot Paper (Controle)", 
        "📊 Freq. Coordenação", 
        "📦 Espaço-Temporal", 
        "🎥 Animações 3D",
        "📊 Tabela Completa"
    ])

    with tab_cin:
        st.subheader("📈 Curvas Cinemáticas Normalizadas (0-100% do Ciclo)")
        x_axis_perc = np.linspace(0, 100, 101)
        sub_t1, sub_t2, sub_t3, sub_t4 = st.tabs(["🟢 Normativo (Controle)", "⚖️ Comparação Articular", "⚖️ Comparação Segmentar", "🔍 Curvas Individuais"])

        with sub_t1:
            st.markdown("<h5 style='text-align:center;'>Média Bilateral Isolada - Grupo Controle</h5>", unsafe_allow_html=True)
            grupo_controle = [g for g in grupos_estudo if 'control' in g.lower()]
            if grupo_controle:
                grp_ctrl = grupo_controle[0]
                fig_ctrl, axs_ctrl = plt.subplots(2, 3, figsize=(15, 8))
                articulacoes = ['Quad', 'Joel', 'Torn']
                titulos_art = ['Hip angular rotation (°)', 'Knee angular rotation (°)', 'Ankle angular rotation (°)']
                for i, art in enumerate(articulacoes):
                    ax = axs_ctrl[0, i]; ax.set_title(titulos_art[i], fontweight='bold'); ax.axhline(0, color='black', lw=0.8)
                    if i == 0: ax.set_ylabel("Angular rotation (°)")
                    ciclos = dados_curvas[grp_ctrl]['Art'][f"{art}_D"] + dados_curvas[grp_ctrl]['Art'][f"{art}_E"]
                    if ciclos: ax.plot(x_axis_perc, np.mean(np.array(ciclos), axis=0), color='black', lw=1.5)

                segmentos = ['Coxa', 'Perna', 'Pe']
                titulos_seg = ['Thigh angular rotation (°)', 'Shank angular rotation (°)', 'Foot angular rotation (°)']
                for i, seg in enumerate(segmentos):
                    ax = axs_ctrl[1, i]; ax.set_xlabel("% Cycle"); ax.set_title(titulos_seg[i], fontweight='bold'); ax.axhline(0, color='black', lw=0.8)
                    if i == 0: ax.set_ylabel("Angular rotation (°)")
                    ciclos = dados_curvas[grp_ctrl]['Seg'][f"{seg}_D"] + dados_curvas[grp_ctrl]['Seg'][f"{seg}_E"]
                    if ciclos: ax.plot(x_axis_perc, np.mean(np.array(ciclos), axis=0), color='black', lw=1.5)
                plt.tight_layout(); st.pyplot(fig_ctrl); plt.close(fig_ctrl)

        with sub_t2:
            st.markdown("<h5 style='text-align:center;'>Comparativo Articular Bilateral</h5>", unsafe_allow_html=True)
            fig_comp_art, axs_comp_art = plt.subplots(1, 3, figsize=(15, 4.5))
            for i, art in enumerate(['Quad', 'Joel', 'Torn']):
                ax = axs_comp_art[i]; ax.set_title(['Hip', 'Knee', 'Ankle'][i], fontweight='bold'); ax.set_xlabel("% Cycle"); ax.axhline(0, color='black', lw=0.8)
                if i == 0: ax.set_ylabel("Angular rotation (°)")
                for idx, grp in enumerate(grupos_estudo):
                    ciclos = dados_curvas[grp]['Art'][f"{art}_D"] + dados_curvas[grp]['Art'][f"{art}_E"]
                    if ciclos:
                        cor, ls, lw = obter_estilo(grp, idx)
                        ax.plot(x_axis_perc, np.mean(np.array(ciclos), axis=0), label=grp, color=cor, linestyle=ls, lw=lw)
                if i == 2: ax.legend(loc='best', frameon=False)
            plt.tight_layout(); st.pyplot(fig_comp_art); plt.close(fig_comp_art)

        with sub_t3:
            st.markdown("<h5 style='text-align:center;'>Comparativo Segmentar Bilateral</h5>", unsafe_allow_html=True)
            fig_comp_seg, axs_comp_seg = plt.subplots(1, 3, figsize=(15, 4.5))
            for i, seg in enumerate(['Coxa', 'Perna', 'Pe']):
                ax = axs_comp_seg[i]; ax.set_title(['Thigh angular rotation (°)', 'Shank angular rotation (°)', 'Foot angular rotation (°)'][i], fontweight='bold'); ax.set_xlabel("% Cycle"); ax.axhline(0, color='black', lw=0.8)
                if i == 0: ax.set_ylabel("Angular rotation (°)")
                for idx, grp in enumerate(grupos_estudo):
                    ciclos = dados_curvas[grp]['Seg'][f"{seg}_D"] + dados_curvas[grp]['Seg'][f"{seg}_E"]
                    if ciclos:
                        cor, ls, lw = obter_estilo(grp, idx)
                        ax.plot(x_axis_perc, np.mean(np.array(ciclos), axis=0), label=grp, color=cor, linestyle=ls, lw=lw)
                if i == 2: ax.legend(loc='best', frameon=False)
            plt.tight_layout(); st.pyplot(fig_comp_seg); plt.close(fig_comp_seg)

        with sub_t4:
            cols_sep = st.columns(len(grupos_estudo))
            for idx, grp in enumerate(grupos_estudo):
                with cols_sep[idx]:
                    st.markdown(f"<h5 style='text-align:center;'>Grupo: {grp}</h5>", unsafe_allow_html=True)
                    fig_ind, axs_ind = plt.subplots(6, 2, figsize=(7, 18), sharex=True)
                    map_ind = [('Quad_D', 0, 0, 'Art'), ('Quad_E', 0, 1, 'Art'), ('Joel_D', 1, 0, 'Art'), ('Joel_E', 1, 1, 'Art'),
                               ('Torn_D', 2, 0, 'Art'), ('Torn_E', 2, 1, 'Art'), ('Coxa_D', 3, 0, 'Seg'), ('Coxa_E', 3, 1, 'Seg'),
                               ('Perna_D', 4, 0, 'Seg'), ('Perna_E', 4, 1, 'Seg'), ('Pe_D', 5, 0, 'Seg'), ('Pe_E', 5, 1, 'Seg')]
                    cor, ls, lw = obter_estilo(grp, idx)
                    for chave, row, col, tipo in map_ind:
                        ax = axs_ind[row, col]; ax.set_title(chave, fontweight='bold', fontsize=10)
                        if row == 5: ax.set_xlabel("% Cycle", fontsize=9)
                        ciclos = dados_curvas[grp][tipo][chave]
                        if ciclos: ax.plot(x_axis_perc, np.mean(ciclos, axis=0), color=cor, linestyle=ls, lw=lw); ax.axhline(0, color='black', lw=0.8)
                    plt.tight_layout(); st.pyplot(fig_ind); plt.close(fig_ind)

    with tab_aa:
        st.subheader("🔄 Diagramas Angle-Angle (Ciclogramas Espaciais)")
        st.markdown("Padrão visual adaptado para artigos científicos. A marca preta indica o Contato Inicial (0%) e o cruzamento das linhas orienta o centro (0°).")
        
        sub_aa1, sub_aa2, sub_aa3, sub_aa4 = st.tabs(["🟢 Padrão Normativo (Controle)", "⚖️ Comparação Articular", "⚖️ Comparação Segmentar", "🔍 Curvas Individuais por Lado"])

        with sub_aa1:
            st.markdown("<h5 style='text-align:center;'>Média Bilateral Isolada - Grupo Controle</h5>", unsafe_allow_html=True)
            grupo_controle = [g for g in grupos_estudo if 'control' in g.lower()]
            if grupo_controle:
                grp = grupo_controle[0]
                fig_aa_ctrl, axs_aa_ctrl = plt.subplots(2, 2, figsize=(9, 9))
                pares_norm_aa = [(axs_aa_ctrl[0,0], 'Quad', 'Joel', 'Art', 'Hip (°)', 'Knee (°)'), (axs_aa_ctrl[0,1], 'Joel', 'Torn', 'Art', 'Knee (°)', 'Ankle (°)'),
                                 (axs_aa_ctrl[1,0], 'Coxa', 'Perna', 'Seg', 'Thigh (°)', 'Shank (°)'), (axs_aa_ctrl[1,1], 'Perna', 'Pe', 'Seg', 'Shank (°)', 'Foot (°)')]
                for ax, x_k, y_k, tipo, label_x, label_y in pares_norm_aa:
                    x_cics = dados_curvas[grp][tipo][f"{x_k}_D"] + dados_curvas[grp][tipo][f"{x_k}_E"]
                    y_cics = dados_curvas[grp][tipo][f"{y_k}_D"] + dados_curvas[grp][tipo][f"{y_k}_E"]
                    if x_cics and y_cics:
                        x_mean, y_mean = np.mean(np.array(x_cics), axis=0), np.mean(np.array(y_cics), axis=0)
                        ax.plot(x_mean, y_mean, color='black', lw=1.5); ax.scatter(x_mean[0], y_mean[0], color='black', s=40, zorder=5)
                    ax.set_xlabel(label_x, fontsize=10); ax.set_ylabel(label_y, fontsize=10); ax.axhline(0, color='black', lw=0.5); ax.axvline(0, color='black', lw=0.5)
                plt.tight_layout(); st.pyplot(fig_aa_ctrl); plt.close(fig_aa_ctrl)
            else:
                st.info("⚠️ Nenhum grupo com o termo 'Controle' foi detectado.")

        with sub_aa2:
            st.markdown("<h5 style='text-align:center;'>Comparativo Articular Bilateral</h5>", unsafe_allow_html=True)
            fig_aa_art, axs_aa_art = plt.subplots(1, 2, figsize=(10, 5))
            pares_comp_art = [(axs_aa_art[0], 'Quad', 'Joel', 'Hip (°)', 'Knee (°)'), (axs_aa_art[1], 'Joel', 'Torn', 'Knee (°)', 'Ankle (°)')]
            for ax, x_k, y_k, label_x, label_y in pares_comp_art:
                ax.set_xlabel(label_x); ax.set_ylabel(label_y); ax.axhline(0, color='black', lw=0.5); ax.axvline(0, color='black', lw=0.5)
                for idx, grp in enumerate(grupos_estudo):
                    x_cics = dados_curvas[grp]['Art'][f"{x_k}_D"] + dados_curvas[grp]['Art'][f"{x_k}_E"]
                    y_cics = dados_curvas[grp]['Art'][f"{y_k}_D"] + dados_curvas[grp]['Art'][f"{y_k}_E"]
                    if x_cics and y_cics:
                        cor, ls, lw = obter_estilo(grp, idx)
                        ax.plot(np.mean(x_cics, axis=0), np.mean(y_cics, axis=0), label=grp, color=cor, linestyle=ls, lw=1.5)
                ax.legend(loc='best', frameon=False)
            plt.tight_layout(); st.pyplot(fig_aa_art); plt.close(fig_aa_art)

        with sub_aa3:
            st.markdown("<h5 style='text-align:center;'>Comparativo Segmentar Bilateral</h5>", unsafe_allow_html=True)
            fig_aa_seg, axs_aa_seg = plt.subplots(1, 2, figsize=(10, 5))
            pares_comp_seg = [(axs_aa_seg[0], 'Coxa', 'Perna', 'Thigh (°)', 'Shank (°)'), (axs_aa_seg[1], 'Perna', 'Pe', 'Shank (°)', 'Foot (°)')]
            for ax, x_k, y_k, label_x, label_y in pares_comp_seg:
                ax.set_xlabel(label_x); ax.set_ylabel(label_y); ax.axhline(0, color='black', lw=0.5); ax.axvline(0, color='black', lw=0.5)
                for idx, grp in enumerate(grupos_estudo):
                    x_cics = dados_curvas[grp]['Seg'][f"{x_k}_D"] + dados_curvas[grp]['Seg'][f"{x_k}_E"]
                    y_cics = dados_curvas[grp]['Seg'][f"{y_k}_D"] + dados_curvas[grp]['Seg'][f"{y_k}_E"]
                    if x_cics and y_cics:
                        cor, ls, lw = obter_estilo(grp, idx)
                        ax.plot(np.mean(x_cics, axis=0), np.mean(y_cics, axis=0), label=grp, color=cor, linestyle=ls, lw=1.5)
                ax.legend(loc='best', frameon=False)
            plt.tight_layout(); st.pyplot(fig_aa_seg); plt.close(fig_aa_seg)

        with sub_aa4:
            cols_ind_aa = st.columns(len(grupos_estudo))
            for idx, grp in enumerate(grupos_estudo):
                with cols_ind_aa[idx]:
                    st.markdown(f"<h5 style='text-align:center;'>Grupo: {grp}</h5>", unsafe_allow_html=True)
                    fig_ind_aa, axs_ind_aa = plt.subplots(4, 2, figsize=(7, 14))
                    mapeamento = [(axs_ind_aa[0,0], 'Quad_D', 'Joel_D', 'Art', 'Hip(°)', 'Knee(°) (D)'), (axs_ind_aa[0,1], 'Quad_E', 'Joel_E', 'Art', 'Hip(°)', 'Knee(°) (E)'),
                                  (axs_ind_aa[1,0], 'Joel_D', 'Torn_D', 'Art', 'Knee(°)', 'Ankle(°) (D)'), (axs_ind_aa[1,1], 'Joel_E', 'Torn_E', 'Art', 'Knee(°)', 'Ankle(°) (E)'),
                                  (axs_ind_aa[2,0], 'Coxa_D', 'Perna_D', 'Seg', 'Thigh(°)', 'Shank(°) (D)'), (axs_ind_aa[2,1], 'Coxa_E', 'Perna_E', 'Seg', 'Thigh(°)', 'Shank(°) (E)'),
                                  (axs_ind_aa[3,0], 'Perna_D', 'Pe_D', 'Seg', 'Shank(°)', 'Foot(°) (D)'), (axs_ind_aa[3,1], 'Perna_E', 'Pe_E', 'Seg', 'Shank(°)', 'Foot(°) (E)')]
                    cor, ls, lw = obter_estilo(grp, idx)
                    for ax, x_k, y_k, tipo, lx, ly in mapeamento:
                        x_cics, y_cics = dados_curvas[grp][tipo][x_k], dados_curvas[grp][tipo][y_k]
                        if x_cics and y_cics: ax.plot(np.mean(x_cics, axis=0), np.mean(y_cics, axis=0), color=cor, linestyle=ls, lw=1.5)
                        ax.set_xlabel(lx, fontsize=9); ax.set_ylabel(ly, fontsize=9); ax.axhline(0, color='black', lw=0.5); ax.axvline(0, color='black', lw=0.5)
                    plt.tight_layout(); st.pyplot(fig_ind_aa); plt.close(fig_ind_aa)

    with tab_ca:
        st.subheader("📈 Coupling Angle - Séries Temporais (Média Normalizada a 100%)")
        sub_ca1, sub_ca2, sub_ca3, sub_ca4 = st.tabs(["🟢 Normativo (Controle)", "⚖️ Comparação Articular", "⚖️ Comparação Segmentar", "🔍 Curvas Individuais"])

        with sub_ca1:
            st.markdown("<h5 style='text-align:center;'>Média Bilateral Isolada - Grupo Controle</h5>", unsafe_allow_html=True)
            grupo_controle = [g for g in grupos_estudo if 'control' in g.lower()]
            if grupo_controle:
                grp_ctrl = grupo_controle[0]
                x_tempo = np.linspace(0, 100, 101)
                
                fig_ca_ctrl, axs_ca_ctrl = plt.subplots(2, 2, figsize=(10, 8))
                pares_ca_ctrl = [(axs_ca_ctrl[0,0], 'Quad_Joel', 'CA_Art', 'Hip-knee coupling angle (°)'), 
                                 (axs_ca_ctrl[0,1], 'Joel_Torn', 'CA_Art', 'Knee-ankle coupling angle (°)'),
                                 (axs_ca_ctrl[1,0], 'Coxa_Perna', 'CA_Seg', 'Thigh-shank coupling angle (°)'), 
                                 (axs_ca_ctrl[1,1], 'Perna_Pe', 'CA_Seg', 'Shank-foot coupling angle (°)')]
                
                for ax, par, tipo, titulo in pares_ca_ctrl:
                    ax.set_ylabel(titulo); ax.set_xlabel("% Cycle")
                    ax.set_ylim(0, 360); ax.set_yticks([0, 180, 360])
                    ciclos = dados_curvas[grp_ctrl][tipo][f"{par}_D"] + dados_curvas[grp_ctrl][tipo][f"{par}_E"]
                    if ciclos: ax.plot(x_tempo, media_circular(ciclos), color='black', lw=1.5)
                plt.tight_layout(); st.pyplot(fig_ca_ctrl); plt.close(fig_ca_ctrl)

        with sub_ca2:
            st.markdown("<h5 style='text-align:center;'>Comparativo Articular Bilateral (CA)</h5>", unsafe_allow_html=True)
            fig_ca_art, axs_ca_art = plt.subplots(1, 2, figsize=(10, 4.5))
            for idx_par, (ax, par, titulo) in enumerate([(axs_ca_art[0], 'Quad_Joel', 'Hip-knee (°)' ), (axs_ca_art[1], 'Joel_Torn', 'Knee-ankle (°)')]):
                ax.set_ylabel(titulo); ax.set_xlabel("% Cycle")
                ax.set_ylim(0, 360); ax.set_yticks([0, 180, 360])
                for idx, grp in enumerate(grupos_estudo):
                    x_tempo = np.linspace(0, 100, 101)
                    ciclos = dados_curvas[grp]['CA_Art'][f"{par}_D"] + dados_curvas[grp]['CA_Art'][f"{par}_E"]
                    if ciclos:
                        cor, ls, lw = obter_estilo(grp, idx)
                        ax.plot(x_tempo, media_circular(ciclos), label=grp, color=cor, linestyle=ls, lw=lw)
                if idx_par == 1: ax.legend(loc='best', frameon=False)
            plt.tight_layout(); st.pyplot(fig_ca_art); plt.close(fig_ca_art)

        with sub_ca3:
            st.markdown("<h5 style='text-align:center;'>Comparativo Segmentar Bilateral (CA)</h5>", unsafe_allow_html=True)
            fig_ca_seg, axs_ca_seg = plt.subplots(1, 2, figsize=(10, 4.5))
            for idx_par, (ax, par, titulo) in enumerate([(axs_ca_seg[0], 'Coxa_Perna', 'Thigh-shank (°)'), (axs_ca_seg[1], 'Perna_Pe', 'Shank-foot (°)')]):
                ax.set_ylabel(titulo); ax.set_xlabel("% Cycle")
                ax.set_ylim(0, 360); ax.set_yticks([0, 180, 360])
                for idx, grp in enumerate(grupos_estudo):
                    x_tempo = np.linspace(0, 100, 101)
                    ciclos = dados_curvas[grp]['CA_Seg'][f"{par}_D"] + dados_curvas[grp]['CA_Seg'][f"{par}_E"]
                    if ciclos:
                        cor, ls, lw = obter_estilo(grp, idx)
                        ax.plot(x_tempo, media_circular(ciclos), label=grp, color=cor, linestyle=ls, lw=lw)
                if idx_par == 1: ax.legend(loc='best', frameon=False)
            plt.tight_layout(); st.pyplot(fig_ca_seg); plt.close(fig_ca_seg)

        with sub_ca4:
            cols_sep_ca = st.columns(len(grupos_estudo))
            for idx, grp in enumerate(grupos_estudo):
                with cols_sep_ca[idx]:
                    st.markdown(f"<h5 style='text-align:center;'>Grupo: {grp}</h5>", unsafe_allow_html=True)
                    x_tempo = np.linspace(0, 100, 101)
                    fig_ind_ca, axs_ind_ca = plt.subplots(4, 2, figsize=(7, 12), sharex=True)
                    map_ca_ind = [('Quad_Joel_D', 0, 0, 'CA_Art'), ('Quad_Joel_E', 0, 1, 'CA_Art'), ('Joel_Torn_D', 1, 0, 'CA_Art'), ('Joel_Torn_E', 1, 1, 'CA_Art'),
                                  ('Coxa_Perna_D', 2, 0, 'CA_Seg'), ('Coxa_Perna_E', 2, 1, 'CA_Seg'), ('Perna_Pe_D', 3, 0, 'CA_Seg'), ('Perna_Pe_E', 3, 1, 'CA_Seg')]
                    cor, ls, lw = obter_estilo(grp, idx)
                    for chave, row, col, tipo in map_ca_ind:
                        ax = axs_ind_ca[row, col]; ax.set_title(chave, fontweight='bold', fontsize=10)
                        if col == 0: ax.set_ylabel("CA (°)", fontsize=9)
                        if row == 3: ax.set_xlabel("% Cycle", fontsize=9)
                        ax.set_ylim(0, 360); ax.set_yticks([0, 180, 360])
                        ciclos = dados_curvas[grp][tipo][chave]
                        if ciclos: ax.plot(x_tempo, media_circular(ciclos), color=cor, linestyle=ls, lw=lw)
                    plt.tight_layout(); st.pyplot(fig_ind_ca); plt.close(fig_ind_ca)

    with tab_paper:
        st.subheader("📄 Plot Paper: Padrão de Normalidade (Grupo Controle)")
        st.info("Visão completa (Cinemática, Angle-Angle e Coupling Angle) baseada na média bilateral do grupo Controle.")
        
        grupo_controle = [g for g in grupos_estudo if 'control' in g.lower()]
        
        if not grupo_controle:
            st.warning("⚠️ Nenhum grupo com o termo 'Controle' foi detectado nos arquivos processados.")
        else:
            grp_ctrl = grupo_controle[0]
            sub_paper_seg, sub_paper_art = st.tabs(["📐 Segmentar", "📍 Articular"])
            x_perc = np.linspace(0, 100, 101)
            
            with sub_paper_seg:
                fig_pseg = plt.figure(figsize=(14, 12))
                axA = plt.subplot2grid((3, 6), (0, 0), colspan=2); axB = plt.subplot2grid((3, 6), (0, 2), colspan=2); axC = plt.subplot2grid((3, 6), (0, 4), colspan=2)
                axD = plt.subplot2grid((3, 6), (1, 0), colspan=3); axE = plt.subplot2grid((3, 6), (1, 3), colspan=3)
                axF = plt.subplot2grid((3, 6), (2, 0), colspan=3); axG = plt.subplot2grid((3, 6), (2, 3), colspan=3)
                
                cics_coxa = dados_curvas[grp_ctrl]['Seg']['Coxa_D'] + dados_curvas[grp_ctrl]['Seg']['Coxa_E']
                cics_perna = dados_curvas[grp_ctrl]['Seg']['Perna_D'] + dados_curvas[grp_ctrl]['Seg']['Perna_E']
                cics_pe = dados_curvas[grp_ctrl]['Seg']['Pe_D'] + dados_curvas[grp_ctrl]['Seg']['Pe_E']
                
                if cics_coxa and cics_perna and cics_pe:
                    coxa = np.mean(cics_coxa, axis=0); perna = np.mean(cics_perna, axis=0); pe = np.mean(cics_pe, axis=0)
                    
                    axA.plot(x_perc, coxa, 'k-', lw=1.5); axA.set_title('Thigh angular rotation (°)', fontweight='bold'); axA.set_xlabel('% Cycle'); axA.axhline(0, color='black', lw=0.5)
                    axB.plot(x_perc, perna, 'k-', lw=1.5); axB.set_title('Shank angular rotation (°)', fontweight='bold'); axB.set_xlabel('% Cycle'); axB.axhline(0, color='black', lw=0.5)
                    axC.plot(x_perc, pe, 'k-', lw=1.5); axC.set_title('Foot angular rotation (°)', fontweight='bold'); axC.set_xlabel('% Cycle'); axC.axhline(0, color='black', lw=0.5)
                    
                    axD.plot(coxa, perna, 'k-', lw=1.5); axD.scatter(coxa[0], perna[0], color='black', s=40, zorder=5); axD.set_title('Thigh-Shank Angle-Angle', fontweight='bold'); axD.set_xlabel('Thigh (°)'); axD.set_ylabel('Shank (°)'); axD.axhline(0, color='black', lw=0.5); axD.axvline(0, color='black', lw=0.5)
                    axE.plot(perna, pe, 'k-', lw=1.5); axE.scatter(perna[0], pe[0], color='black', s=40, zorder=5); axE.set_title('Shank-Foot Angle-Angle', fontweight='bold'); axE.set_xlabel('Shank (°)'); axE.set_ylabel('Foot (°)'); axE.axhline(0, color='black', lw=0.5); axE.axvline(0, color='black', lw=0.5)
                    
                    ca_cp = media_circular(dados_curvas[grp_ctrl]['CA_Seg']['Coxa_Perna_D'] + dados_curvas[grp_ctrl]['CA_Seg']['Coxa_Perna_E'])
                    ca_pp = media_circular(dados_curvas[grp_ctrl]['CA_Seg']['Perna_Pe_D'] + dados_curvas[grp_ctrl]['CA_Seg']['Perna_Pe_E'])
                    
                    axF.plot(x_perc, ca_cp, 'k-', lw=1.5); axF.set_title('Thigh-shank coupling angle (°)', fontweight='bold'); axF.set_xlabel('% Cycle'); axF.set_ylim(0, 360); axF.set_yticks([0, 180, 360])
                    axG.plot(x_perc, ca_pp, 'k-', lw=1.5); axG.set_title('Shank-foot coupling angle (°)', fontweight='bold'); axG.set_xlabel('% Cycle'); axG.set_ylim(0, 360); axG.set_yticks([0, 180, 360])
                    
                plt.tight_layout(); st.pyplot(fig_pseg); plt.close(fig_pseg)

            with sub_paper_art:
                fig_part = plt.figure(figsize=(14, 12))
                axA = plt.subplot2grid((3, 6), (0, 0), colspan=2); axB = plt.subplot2grid((3, 6), (0, 2), colspan=2); axC = plt.subplot2grid((3, 6), (0, 4), colspan=2)
                axD = plt.subplot2grid((3, 6), (1, 0), colspan=3); axE = plt.subplot2grid((3, 6), (1, 3), colspan=3)
                axF = plt.subplot2grid((3, 6), (2, 0), colspan=3); axG = plt.subplot2grid((3, 6), (2, 3), colspan=3)
                
                cics_quad = dados_curvas[grp_ctrl]['Art']['Quad_D'] + dados_curvas[grp_ctrl]['Art']['Quad_E']
                cics_joel = dados_curvas[grp_ctrl]['Art']['Joel_D'] + dados_curvas[grp_ctrl]['Art']['Joel_E']
                cics_torn = dados_curvas[grp_ctrl]['Art']['Torn_D'] + dados_curvas[grp_ctrl]['Art']['Torn_E']
                
                if cics_quad and cics_joel and cics_torn:
                    quad = np.mean(cics_quad, axis=0); joel = np.mean(cics_joel, axis=0); torn = np.mean(cics_torn, axis=0)
                    
                    axA.plot(x_perc, quad, 'k-', lw=1.5); axA.set_title('Hip angular rotation (°)', fontweight='bold'); axA.set_xlabel('% Cycle'); axA.axhline(0, color='black', lw=0.5)
                    axB.plot(x_perc, joel, 'k-', lw=1.5); axB.set_title('Knee angular rotation (°)', fontweight='bold'); axB.set_xlabel('% Cycle'); axB.axhline(0, color='black', lw=0.5)
                    axC.plot(x_perc, torn, 'k-', lw=1.5); axC.set_title('Ankle angular rotation (°)', fontweight='bold'); axC.set_xlabel('% Cycle'); axC.axhline(0, color='black', lw=0.5)
                    
                    axD.plot(quad, joel, 'k-', lw=1.5); axD.scatter(quad[0], joel[0], color='black', s=40, zorder=5); axD.set_title('Hip-Knee Angle-Angle', fontweight='bold'); axD.set_xlabel('Hip (°)'); axD.set_ylabel('Knee (°)'); axD.axhline(0, color='black', lw=0.5); axD.axvline(0, color='black', lw=0.5)
                    axE.plot(joel, torn, 'k-', lw=1.5); axE.scatter(joel[0], torn[0], color='black', s=40, zorder=5); axE.set_title('Knee-Ankle Angle-Angle', fontweight='bold'); axE.set_xlabel('Knee (°)'); axE.set_ylabel('Ankle (°)'); axE.axhline(0, color='black', lw=0.5); axE.axvline(0, color='black', lw=0.5)
                    
                    ca_qj = media_circular(dados_curvas[grp_ctrl]['CA_Art']['Quad_Joel_D'] + dados_curvas[grp_ctrl]['CA_Art']['Quad_Joel_E'])
                    ca_jt = media_circular(dados_curvas[grp_ctrl]['CA_Art']['Joel_Torn_D'] + dados_curvas[grp_ctrl]['CA_Art']['Joel_Torn_E'])
                    
                    axF.plot(x_perc, ca_qj, 'k-', lw=1.5); axF.set_title('Hip-knee coupling angle (°)', fontweight='bold'); axF.set_xlabel('% Cycle'); axF.set_ylim(0, 360); axF.set_yticks([0, 180, 360])
                    axG.plot(x_perc, ca_jt, 'k-', lw=1.5); axG.set_title('Knee-ankle coupling angle (°)', fontweight='bold'); axG.set_xlabel('% Cycle'); axG.set_ylim(0, 360); axG.set_yticks([0, 180, 360])
                    
                plt.tight_layout(); st.pyplot(fig_part); plt.close(fig_part)

    with tab_freq:
        st.subheader("📊 Frequência de Coordenação Vetorial (Vector Coding)")
        
        bw_padroes = {
            'Proximal': {'cor': '#ffffff', 'hatch': '////'},
            'EmFase':   {'cor': '#cccccc', 'hatch': '\\\\\\\\'},
            'Distal':   {'cor': '#777777', 'hatch': '....'},
            'AntiFase': {'cor': '#222222', 'hatch': ''}
        }
        
        tipo_coord = st.radio("Selecione o Modelo de Coordenação para os Histogramas:", ["📐 Segmentar (Coxa-Perna / Perna-Pé)", "📍 Articular (Quad-Joel / Joel-Torn)"], horizontal=True)
        
        if "Segmentar" in tipo_coord:
            pares_map = [('Coxa_Perna_D', 'Coxa-Perna (DIR)'), ('Coxa_Perna_E', 'Coxa-Perna (ESQ)'), ('Perna_Pe_D', 'Perna-Pé (DIR)'), ('Perna_Pe_E', 'Perna-Pé (ESQ)')]
            pares_labels = ['Coxa_Perna', 'Perna_Pe']
            pares_nomes = ['Thigh-Shank', 'Shank-Foot']
        else:
            pares_map = [('Quad_Joel_D', 'Quad-Joel (DIR)'), ('Quad_Joel_E', 'Quad-Joel (ESQ)'), ('Joel_Torn_D', 'Joel-Torn (DIR)'), ('Joel_Torn_E', 'Joel-Torn (ESQ)')]
            pares_labels = ['Quad_Joel', 'Joel_Torn']
            pares_nomes = ['Hip-Knee', 'Knee-Ankle']

        def compilar_frequencia_bilateral_dp(grupos, pares_origem, fase="Apoio"):
            dados = {g: {k: {'media': 0, 'dp': 0} for k in bw_padroes} for g in grupos}
            for grp in grupos:
                valores_grp = {k: [] for k in bw_padroes}
                for p in [proc for proc in st.session_state.processadores if proc.grupo == grp]:
                    m_prox, m_fase, m_dist, m_anti, contagem = 0, 0, 0, 0, 0
                    for par_chave in pares_origem:
                        lado = par_chave.split('_')[-1]
                        pct_apoio = p.fases_marcha.get(lado, {}).get('Apoio', 60.0)
                        idx_apoio = int(round(pct_apoio)) if not np.isnan(pct_apoio) else 60
                        
                        serie_completa = p.coord_vetorial_series.get(par_chave, [])
                        if fase == "Apoio": fatia = serie_completa[:idx_apoio]
                        else: fatia = serie_completa[idx_apoio:]
                            
                        fatia = [x for x in fatia if x != 'Ruido']
                        total = len(fatia)
                        if total > 0:
                            m_prox += fatia.count('Proximal')/total; m_fase += fatia.count('EmFase')/total
                            m_dist += fatia.count('Distal')/total; m_anti += fatia.count('AntiFase')/total
                            contagem += 1
                    if contagem > 0:
                        valores_grp['Proximal'].append((m_prox/contagem)*100)
                        valores_grp['EmFase'].append((m_fase/contagem)*100)
                        valores_grp['Distal'].append((m_dist/contagem)*100)
                        valores_grp['AntiFase'].append((m_anti/contagem)*100)
                
                for k in bw_padroes:
                    dados[grp][k]['media'] = np.mean(valores_grp[k]) if len(valores_grp[k]) > 0 else 0
                    dados[grp][k]['dp'] = np.std(valores_grp[k]) if len(valores_grp[k]) > 0 else 0
            return dados

        st.markdown("### Frequência dos Modos de Coordenação (Média Bilateral com Desvio Padrão)")
        sub_freq_apoio, sub_freq_balanco = st.tabs(["🦵 Fase de Apoio", "✈️ Fase de Balanço"])
        
        def plotar_histograma_p_b(ax, dados_comp, titulo):
            grps = list(dados_comp.keys())
            x = np.arange(len(grps))
            width = 0.5
            
            bottom = np.zeros(len(grps))
            for padrao, estilo in bw_padroes.items():
                medias = [dados_comp[g][padrao]['media'] for g in grps]
                dps = [dados_comp[g][padrao]['dp'] for g in grps]
                bars = ax.bar(x, medias, width, bottom=bottom, label=padrao, color=estilo['cor'], edgecolor='black', hatch=estilo['hatch'])
                for i, bar in enumerate(bars):
                    h = bar.get_height()
                    if h > 4.0: 
                        texto = f"{medias[i]:.1f}% (±{dps[i]:.1f})"
                        ax.text(bar.get_x() + bar.get_width() + 0.05, bar.get_y() + h/2, texto, ha='left', va='center', color='black', fontweight='bold', fontsize=8)
                bottom += np.array(medias)

            ax.set_title(titulo, fontweight='bold', fontsize=12)
            ax.set_ylabel('Frequency (%)', fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(grps, fontweight='bold', fontsize=10)
            ax.set_ylim(0, 105); ax.set_xlim(-0.5, len(grps) + 0.5) 

        with sub_freq_apoio:
            fig_ap, axs_ap = plt.subplots(1, 2, figsize=(14, 5))
            for i, (p_label, p_nome) in enumerate(zip(pares_labels, pares_nomes)):
                dados_ap = compilar_frequencia_bilateral_dp(grupos_estudo, [f"{p_label}_D", f"{p_label}_E"], fase="Apoio")
                plotar_histograma_p_b(axs_ap[i], dados_ap, f"{p_nome} (Stance)")
            axs_ap[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Modos", frameon=False)
            plt.tight_layout(); st.pyplot(fig_ap); plt.close(fig_ap)

        with sub_freq_balanco:
            fig_bal, axs_bal = plt.subplots(1, 2, figsize=(14, 5))
            for i, (p_label, p_nome) in enumerate(zip(pares_labels, pares_nomes)):
                dados_bal = compilar_frequencia_bilateral_dp(grupos_estudo, [f"{p_label}_D", f"{p_label}_E"], fase="Balanco")
                plotar_histograma_p_b(axs_bal[i], dados_bal, f"{p_nome} (Swing)")
            axs_bal[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Modos", frameon=False)
            plt.tight_layout(); st.pyplot(fig_bal); plt.close(fig_bal)

    with tab_anim:
        st.subheader("Gerador de GIFs Modulares e Biofeedback Visual")
        st.info("Escolha os arquivos e ative as análises em tempo real que deseja adicionar ao redor do modelo 3D.")
        
        lista_nomes = [p.nome_arq for p in st.session_state.processadores]
        col_selecao, col_vel = st.columns([2, 1])
        with col_selecao: selecionados = st.multiselect("Selecione os arquivos para animar:", lista_nomes)
        with col_vel: vel_opcao = st.radio("Velocidade de Reprodução:", ["100% (Normal)", "75% (Lenta)", "50% (Muito Lenta)"])
        fps_escolhido = {"100% (Normal)": 20, "75% (Lenta)": 15, "50% (Muito Lenta)": 10}[vel_opcao]
        
        st.markdown("---")
        st.markdown("**Opções de Biofeedback em Tempo Real (Adicionados ao GIF):**")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            opt_bussolas = st.checkbox("🧭 Bússolas de Coordenação Vetorial (Membro Dir/Esq) - À Esquerda")
        with col_opt2:
            opt_ang = st.checkbox("📈 Ângulos Segmentares (Membro Direito) - À Direita")
            opt_aa = st.checkbox("🔄 Diagramas Angle-Angle (Membro Direito) - À Direita")
            opt_ca = st.checkbox("🔗 Coupling Angle Série Temporal (Membro Direito) - À Direita")

        st.markdown("---")
        if st.button("Gerar GIFs Selecionados", type="primary"):
            for p in st.session_state.processadores:
                if p.nome_arq in selecionados:
                    with st.spinner(f"Processando animação modular para: {p.nome_arq}..."):
                        fd, tmp_gif_path = tempfile.mkstemp(suffix='.gif'); os.close(fd) 
                        viz = GeradorVisual(
                            p, p.nome_arq, 
                            opt_bussolas=opt_bussolas, opt_ang=opt_ang, opt_aa=opt_aa, opt_ca=opt_ca
                        )
                        sucesso, msg = viz.salvar(tmp_gif_path, step=3, fps_anim=fps_escolhido)
                        if sucesso:
                            st.image(tmp_gif_path, use_container_width=True)
                            with open(tmp_gif_path, "rb") as file_gif: st.download_button(f"📥 Baixar GIF", data=file_gif, file_name=f"{p.nome_arq.split('.')[0]}_3D.gif", mime="image/gif")
                        else: st.error(f"Falha: {msg}")

    with tab_est:
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


    with tab_tab:
        st.subheader("📊 Tabelas de Resultados Médios e Exportação")
        
        if not st.session_state.processadores:
            st.info("Nenhum dado processado ainda. Faça o upload dos arquivos C3D para gerar a tabela.")
        else:
            dados_tabela = []
            
            for p in st.session_state.processadores:
                linha = {
                    'ID_Paciente': p.id_paciente,
                    'Grupo': p.grupo,
                    'Velocidade (m/s)': p.velocidade_media,
                    'Apoio DIR (%)': p.fases_marcha['D'].get('Apoio', np.nan),
                    'Apoio ESQ (%)': p.fases_marcha['E'].get('Apoio', np.nan),
                    'Clearance DIR (mm)': p.foot_clearance.get('D', np.nan),
                    'Clearance ESQ (mm)': p.foot_clearance.get('E', np.nan),
                    'Passo DIR (mm)': p.comprimento_passo.get('D', np.nan),
                    'Passo ESQ (mm)': p.comprimento_passo.get('E', np.nan),
                    'Passo DIR (% Altura)': p.passo_norm.get('D', np.nan) if hasattr(p, 'passo_norm') else np.nan,
                    'Passo ESQ (% Altura)': p.passo_norm.get('E', np.nan) if hasattr(p, 'passo_norm') else np.nan
                }

                # -------------------------------------------------------------
                # EXTRAÇÃO: Ângulos Articulares (Máx, Mín e Delta/Variação)
                # -------------------------------------------------------------
                if hasattr(p, 'angulos_df'):
                    for art in ['Quad_D', 'Quad_E', 'Joel_D', 'Joel_E', 'Torn_D', 'Torn_E']:
                        if art in p.angulos_df.columns:
                            v_max = p.angulos_df[art].max()
                            v_min = p.angulos_df[art].min()
                            linha[f'{art} Máx (°)'] = v_max
                            linha[f'{art} Mín (°)'] = v_min
                            linha[f'{art} Delta (°)'] = v_max - v_min

                # -------------------------------------------------------------
                # EXTRAÇÃO: Ângulos Segmentares (Máx, Mín e Delta/Variação)
                # -------------------------------------------------------------
                if hasattr(p, 'segmentos_df'):
                    for seg in ['Coxa_D', 'Coxa_E', 'Perna_D', 'Perna_E', 'Pe_D', 'Pe_E']:
                        if seg in p.segmentos_df.columns:
                            v_max = p.segmentos_df[seg].max()
                            v_min = p.segmentos_df[seg].min()
                            linha[f'{seg} Máx (°)'] = v_max
                            linha[f'{seg} Mín (°)'] = v_min
                            linha[f'{seg} Delta (°)'] = v_max - v_min

                # -------------------------------------------------------------
                # EXTRAÇÃO: Vector Coding Segmentar (Frequência e CAV)
                # -------------------------------------------------------------
                if hasattr(p, 'coord_vetorial_series') and p.coord_vetorial_series:
                    mapa_pares = {
                        'Coxa_Perna_D': 'Segm_CP_DIR',
                        'Coxa_Perna_E': 'Segm_CP_ESQ',
                        'Perna_Pe_D': 'Segm_PP_DIR',
                        'Perna_Pe_E': 'Segm_PP_ESQ'
                    }
                    
                    for par_interno, par_exportacao in mapa_pares.items():
                        lado = par_interno.split('_')[-1]
                        pct_apoio = p.fases_marcha.get(lado, {}).get('Apoio', 60.0)
                        idx_apoio = int(round(pct_apoio)) if not np.isnan(pct_apoio) else 60
                        
                        # --- 1. Extração de Frequência de Modos ---
                        serie_completa = p.coord_vetorial_series.get(par_interno, [])
                        if len(serie_completa) > 0:
                            fatias = {
                                'APOIO': serie_completa[:idx_apoio],
                                'BALANÇO': serie_completa[idx_apoio:]
                            }
                            for fase_nome, fatia in fatias.items():
                                fatia_limpa = [x for x in fatia if x != 'Ruido']
                                total = len(fatia_limpa)
                                for modo in ['Proximal', 'EmFase', 'Distal', 'AntiFase']:
                                    col_name = f"{fase_nome} {par_exportacao} - {modo} (%)"
                                    linha[col_name] = (fatia_limpa.count(modo) / total) * 100 if total > 0 else np.nan

                        # --- 2. Extração da Variabilidade (Lendo o CAV pré-calculado no processador) ---
                        if hasattr(p, 'coord_vetorial_cav') and par_interno in p.coord_vetorial_cav:
                            cav_continuo = p.coord_vetorial_cav[par_interno]
                            # Usa nanmean para segurança na fase selecionada
                            linha[f'Apoio CAV {par_exportacao} (°)'] = np.nanmean(cav_continuo[:idx_apoio]) if len(cav_continuo[:idx_apoio]) > 0 else np.nan
                            linha[f'Balanço CAV {par_exportacao} (°)'] = np.nanmean(cav_continuo[idx_apoio:]) if len(cav_continuo[idx_apoio:]) > 0 else np.nan
                        else:
                            linha[f'Apoio CAV {par_exportacao} (°)'] = np.nan
                            linha[f'Balanço CAV {par_exportacao} (°)'] = np.nan

                # A LINHA DE OURO QUE HAVIA SE PERDIDO: Anexa os dados extraídos do paciente na tabela!
                dados_tabela.append(linha)

            # Criação do DataFrame Bruto
            df_bruto = pd.DataFrame(dados_tabela)

            # Verificação de segurança estrutural
            if df_bruto.empty or 'Grupo' not in df_bruto.columns or 'ID_Paciente' not in df_bruto.columns:
                st.warning("⚠️ Os dados processados não retornaram variáveis suficientes para tabular. Verifique os arquivos C3D.")
            else:
                # =================================================================
                # DATAFRAME 1: Média por Paciente (Base de Lados Separados)
                # =================================================================
                # Adicionado numeric_only=True como medida de segurança do Pandas 2.0
                df_media_paciente = df_bruto.groupby(['Grupo', 'ID_Paciente']).mean(numeric_only=True).reset_index()
                df_media_paciente = df_media_paciente.round(2)

                st.markdown("### 1. Dados Base (Médias por Membro)")
                st.dataframe(df_media_paciente, use_container_width=True, hide_index=True)

                csv_export_t1 = df_media_paciente.to_csv(index=False, sep=';', decimal=',')
                st.download_button(
                    label="📥 Baixar Tabela 1 - Lados Separados (CSV)",
                    data=csv_export_t1,
                    file_name="GPBIO_Resultados_Por_Lado.csv",
                    mime="text/csv",
                    type="secondary"
                )

                # =================================================================
                # DATAFRAME 2: Média Bilateral (Formato Oficial de Exportação)
                # =================================================================
                st.markdown("---")
                st.markdown("### 2. Tabela Bilateral (Exportação Oficial)")
                
                df_bilateral = pd.DataFrame()
                df_bilateral['Grupo'] = df_media_paciente['Grupo']
                df_bilateral['ID_Paciente'] = df_media_paciente['ID_Paciente']
                
                if 'Velocidade (m/s)' in df_media_paciente.columns:
                    df_bilateral['Velocidade (m/s)'] = df_media_paciente['Velocidade (m/s)']

                # --- Cruzamento Espaço-Temporal ---
                cols_espaco_temporais = [
                    ('Apoio Bilat (%)', 'Apoio DIR (%)', 'Apoio ESQ (%)'),
                    ('Clearance Bilat (mm)', 'Clearance DIR (mm)', 'Clearance ESQ (mm)'),
                    ('Passo Bilat (mm)', 'Passo DIR (mm)', 'Passo ESQ (mm)'),
                    ('Passo Norm Bilat (%)', 'Passo DIR (% Altura)', 'Passo ESQ (% Altura)')
                ]

                for col_nova, col_d, col_e in cols_espaco_temporais:
                    if col_d in df_media_paciente.columns and col_e in df_media_paciente.columns:
                        df_bilateral[col_nova] = df_media_paciente[[col_d, col_e]].mean(axis=1)

                # --- Integralização das Médias Bilaterais de Deltas ---
                deltas_mapear = [
                    ('Quad Delta Bilat (°)', 'Quad_D Delta (°)', 'Quad_E Delta (°)'),
                    ('Joel Delta Bilat (°)', 'Joel_D Delta (°)', 'Joel_E Delta (°)'),
                    ('Torn Delta Bilat (°)', 'Torn_D Delta (°)', 'Torn_E Delta (°)'),
                    ('Coxa Delta Bilat (°)', 'Coxa_D Delta (°)', 'Coxa_E Delta (°)'),
                    ('Perna Delta Bilat (°)', 'Perna_D Delta (°)', 'Perna_E Delta (°)'),
                    ('Pe Delta Bilat (°)', 'Pe_D Delta (°)', 'Pe_E Delta (°)')
                ]

                for col_nova, col_d, col_e in deltas_mapear:
                    if col_d in df_media_paciente.columns and col_e in df_media_paciente.columns:
                        df_bilateral[col_nova] = df_media_paciente[[col_d, col_e]].mean(axis=1)

                # --- Cruzamento Vector Coding Segmentar (Frequência) ---
                modos_vc = ['Proximal (%)', 'EmFase (%)', 'Distal (%)', 'AntiFase (%)']
                pares_vc = [('Segm_CP', 'CP'), ('Segm_PP', 'PP')]

                for original, sigla in pares_vc:
                    for fase_orig, fase_nova in [('APOIO', 'Apoio'), ('BALANÇO', 'Balanço')]:
                        for modo in modos_vc:
                            col_d = f'{fase_orig} {original}_DIR - {modo}'
                            col_e = f'{fase_orig} {original}_ESQ - {modo}'
                            col_nova = f'{fase_nova} {sigla} Bilat - {modo}'
                            
                            if col_d in df_media_paciente.columns and col_e in df_media_paciente.columns:
                                df_bilateral[col_nova] = df_media_paciente[[col_d, col_e]].mean(axis=1)

                # --- Cruzamento Vector Coding Segmentar (CAV / Variabilidade) ---
                for original, sigla in pares_vc:
                    for fase_orig in ['Apoio', 'Balanço']:
                        col_d = f'{fase_orig} CAV {original}_DIR (°)'
                        col_e = f'{fase_orig} CAV {original}_ESQ (°)'
                        col_nova = f'{fase_orig} CAV {sigla} Bilat (°)'
                        
                        if col_d in df_media_paciente.columns and col_e in df_media_paciente.columns:
                            df_bilateral[col_nova] = df_media_paciente[[col_d, col_e]].mean(axis=1)

                # Ajustes estruturais e exibição final
                df_bilateral = df_bilateral.round(2)
                st.dataframe(df_bilateral, use_container_width=True, hide_index=True)

                csv_export_bilateral = df_bilateral.to_csv(index=False, sep=';', decimal=',')
                st.download_button(
                    label="📥 Baixar Tabela Bilateral de Exportação (CSV)",
                    data=csv_export_bilateral,
                    file_name="GPBIO_Resultados_Bilaterais_Export.csv",
                    mime="text/csv",
                    type="primary"
                )
