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
# ENGINE MATEMÁTICA 
# =============================================================================
def vetor(p1, p2): return p2 - p1
def normalizar(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm
def angulo_entre(v1, v2):
    v1_u, v2_u = normalizar(v1), normalizar(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

class ProcessadorCinematico:
    def __init__(self, caminho_arquivo, nome_original, grupo="Geral", df_antropo=None):
        self.caminho = caminho_arquivo
        self.nome_arq = nome_original
        self.grupo=grupo
        self.valido = False
        self.erro_msg = ""

        try:
            self.c3d = ezc3d.c3d(caminho_arquivo)
            self.labels = [l.strip() for l in self.c3d['parameters']['POINT']['LABELS']['value']]
            self.mapa = {lbl: i for i, lbl in enumerate(self.labels)}
            self.freq = self.c3d['parameters']['POINT']['RATE']['value'][0]
            self.dados_raw = self.c3d['data']['points'][:3, :, :]
            self.n_frames = self.dados_raw.shape[2]

            self.dados = self._filtrar_e_inverter()
            self.angulos_df = self._calcular_angulos()
            self.velocidade_media = self._calcular_velocidade_sacrum()
            self.eventos = self.detectar_eventos_zeni()
            self.fases_marcha = self._calcular_fases_marcha()
            self.foot_clearance = self._calcular_foot_clearance()
            self.comprimento_passo = self._calcular_comprimento_passo()
            self.passo_norm = {'D': np.nan, 'E': np.nan}
            if df_antropo is not None:
                nome_limpo = nome_original.lower().replace('.c3d', '')
                id_paciente = nome_limpo.split('_')[0].upper().strip()                
                match = df_antropo[df_antropo['ID'] == id_paciente]                
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
            if not np.isnan(v).any(): return v
        return None

    def _mid(self, n1, n2, f):
        p1, p2 = self._get(n1, f), self._get(n2, f)
        if p1 is not None and p2 is not None: return (p1+p2)/2
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
                if np.isnan(sinal).all(): continue
                
                s_temp = pd.Series(sinal).interpolate(limit_direction='both').bfill().ffill()
                try:
                    filt = signal.filtfilt(b, a, s_temp.to_numpy())
                    out[ax, m, :] = filt
                except: out[ax, m, :] = s_temp
        out[0, :, :] = -1 * out[0, :, :]
        return out

    def _calcular_angulos(self):
        res = {k: [] for k in ['Quad_D','Joel_D','Torn_D','Quad_E','Joel_E','Torn_E']}
        vec_g = np.array([0,0,-1])
        for f in range(self.n_frames):
            h_d = self._get('RIAS',f); k_d = self._mid('RLE','RME',f); a_d = self._mid('RML','RMM',f); p_d = self._mid('RFT1','RFT5',f)
            res['Quad_D'].append(angulo_entre(vetor(h_d, k_d), vec_g) if (h_d is not None and k_d is not None) else np.nan)
            res['Joel_D'].append(angulo_entre(vetor(h_d, k_d), vetor(k_d, a_d)) if (h_d is not None and k_d is not None and a_d is not None) else np.nan)
            res['Torn_D'].append(angulo_entre(vetor(k_d, a_d), vetor(a_d, p_d)) if (k_d is not None and p_d is not None) else np.nan)
            
            h_e = self._get('LIAS',f); k_e = self._mid('LLE','LME',f); a_e = self._mid('LML','LMM',f); p_e = self._mid('LFT1','LFT5',f)
            res['Quad_E'].append(angulo_entre(vetor(h_e, k_e), vec_g) if (h_e is not None and k_e is not None) else np.nan)
            res['Joel_E'].append(angulo_entre(vetor(h_e, k_e), vetor(k_e, a_e)) if (h_e is not None and k_e is not None and a_e is not None) else np.nan)
            res['Torn_E'].append(angulo_entre(vetor(k_e, a_e), vetor(a_e, p_e)) if (k_e is not None and p_e is not None) else np.nan)
        return pd.DataFrame(res)

    def detectar_eventos_zeni(self):
        eventos = {'D': {'HS': [], 'TO': []}, 'E': {'HS': [], 'TO': []}}
        
        rias_data = self._get('RIAS', slice(None))
        lias_data = self._get('LIAS', slice(None))
        
        if rias_data is None or lias_data is None: 
            return eventos 
            
        rias_x = rias_data[0]
        lias_x = lias_data[0]
    
        pelvis_x = (rias_x + lias_x) / 2
        dist_frames = int(self.freq * 0.6)
        
        for lado, cal_label, toe_label in [('D','RCAL','RFT1'), ('E','LCAL','LFT1')]:
            cal_x_data = self._get(cal_label, slice(None))
            toe_x_data = self._get(toe_label, slice(None))
            
            if cal_x_data is None or toe_x_data is None: continue
            
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

    def extrair_ciclos_normalizados(self, vetor_dados, eventos_hs, pontos=101):
        ciclos = []
        if len(eventos_hs) < 2: return []
        for i in range(len(eventos_hs) - 1):
            if eventos_hs[i+1] > len(vetor_dados): continue
            ciclo_bruto = vetor_dados[eventos_hs[i]:eventos_hs[i+1]]
            ciclos.append(np.interp(np.linspace(0, len(ciclo_bruto)-1, pontos), np.arange(len(ciclo_bruto)), ciclo_bruto))
        return ciclos

    def _calcular_coordenacao_vetorial(self):
        res = {par: {'Proximal': np.nan, 'Distal': np.nan, 'EmFase': np.nan, 'AntiFase': np.nan} for par in ['Quad_Joel_D', 'Quad_Joel_E', 'Joel_Torn_D', 'Joel_Torn_E']}
        for lado in ['D', 'E']:
            hss = self.eventos[lado]['HS']
            if len(hss) < 2: continue
            for nome_par, col_prox, col_dist in [(f'Quad_Joel_{lado}', f'Quad_{lado}', f'Joel_{lado}'), (f'Joel_Torn_{lado}', f'Joel_{lado}', f'Torn_{lado}')]:
                c_prox = self.extrair_ciclos_normalizados(self.angulos_df[col_prox].values, hss)
                c_dist = self.extrair_ciclos_normalizados(self.angulos_df[col_dist].values, hss)
                if not c_prox or not c_dist: continue
                
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
        return res

# =============================================================================
# MÓDULO VISUAL (GIFs) 
# =============================================================================
import matplotlib.patches as mpatches

class GeradorVisual:
    def __init__(self, processador, nome_original):
        self.proc = processador
        self.nome_arq = nome_original
        
        self.box = {
            'x': (-1000, 1000), 
            'y': (-1000, 1000), 
            'z': (0, 2000)
            }

    def montar_frame(self, f):
        s = {}
        get = lambda n: self.proc._get(n, f)
        mid = lambda n1, n2: self.proc._mid(n1, n2, f)
        rias, lias = get('RIAS'), get('LIAS'); rips, lips = get('RIPS'), get('LIPS')
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
        if (0 <= a < 22.5) or (337.5 <= a <= 360) or (157.5 <= a < 202.5): return "DOMINÂNCIA PROXIMAL", '#e74c3c'
        elif (22.5 <= a < 67.5) or (202.5 <= a < 247.5): return "EM FASE", '#2ecc71'
        elif (67.5 <= a < 112.5) or (247.5 <= a < 292.5): return "DOMINÂNCIA DISTAL", '#3498db'
        else: return "ANTI-FASE", '#f1c40f'

    def salvar(self, caminho_final, step=3, fps_anim=20):
        fig = plt.figure(figsize=(16, 9))
        
        ax_comp_qj_d = fig.add_axes([0.01, 0.65, 0.15, 0.25])
        ax_comp_jt_d = fig.add_axes([0.01, 0.38, 0.15, 0.25])
        ax_comp_qj_e = fig.add_axes([0.16, 0.65, 0.15, 0.25])
        ax_comp_jt_e = fig.add_axes([0.16, 0.38, 0.15, 0.25])

        ptr_qjd = self._desenhar_fundo_bussola(ax_comp_qj_d, "Quad-Joel (DIR)")
        ptr_jtd = self._desenhar_fundo_bussola(ax_comp_jt_d, "Joel-Torn (DIR)")
        ptr_qje = self._desenhar_fundo_bussola(ax_comp_qj_e, "Quad-Joel (ESQ)")
        ptr_jte = self._desenhar_fundo_bussola(ax_comp_jt_e, "Joel-Torn (ESQ)")

        ax_stats_left = fig.add_axes([0.01, 0.02, 0.30, 0.32]); ax_stats_left.axis('off')
        
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
        
        col_dir = ">> QUADRIL-JOELHO (DIR)\n" + format_f(coord_norm.get('Quad_Joel_D', {})) + "\n\n"
        col_dir += ">> JOELHO-TORNOZELO (DIR)\n" + format_f(coord_norm.get('Joel_Torn_D', {}))
        
        col_esq = ">> QUADRIL-JOELHO (ESQ)\n" + format_f(coord_norm.get('Quad_Joel_E', {})) + "\n\n"
        col_esq += ">> JOELHO-TORNOZELO (ESQ)\n" + format_f(coord_norm.get('Joel_Torn_E', {}))

        ax_stats_left.text(0.00, 0.85, col_dir, va='top', fontsize=9, family='monospace')
        ax_stats_left.text(0.55, 0.85, col_esq, va='top', fontsize=9, family='monospace')

        
        ax_banner.text(0.5, 0.90, "PADRÕES DE COORDENAÇÃO EM TEMPO REAL", ha='center', va='top', fontweight='bold', fontsize=11)
        
      
        ax_banner.text(0.00, 0.50, "Quad-Joel (DIR):", fontweight='bold', fontsize=10)
        ax_banner.text(0.00, 0.15, "Joel-Torn (DIR):", fontweight='bold', fontsize=10)
        txt_qj_d = ax_banner.text(0.24, 0.50, "-", fontweight='bold', fontsize=10)
        txt_jt_d = ax_banner.text(0.24, 0.15, "-", fontweight='bold', fontsize=10)

        
        ax_banner.text(0.53, 0.50, "Quad-Joel (ESQ):", fontweight='bold', fontsize=10)
        ax_banner.text(0.53, 0.15, "Joel-Torn (ESQ):", fontweight='bold', fontsize=10)
        txt_qj_e = ax_banner.text(0.77, 0.50, "-", fontweight='bold', fontsize=10)
        txt_jt_e = ax_banner.text(0.77, 0.15, "-", fontweight='bold', fontsize=10)
        

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
                else: linhas[n], = ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]], c=c, lw=2)

            row = self.proc.angulos_df.iloc[i]
            info = "DADOS ARTICULARES\n" + "="*17 + "\n\n"
            for l, l_full in [('D', 'DIREITO (Vermelho)'), ('E', 'ESQUERDO (Azul)')]:
                info += f">>> LADO {l_full}\n\n"
                for j, j_full in [('Quad', 'Quadril'), ('Joel', 'Joelho'), ('Torn', 'Tornozelo')]:
                    s = stats_ang.get(f'{j}_{l}', {'min':0, 'max':0})
                    info += f"{j_full}:\n"
                    info += f"  Atual: {row[f'{j}_{l}']:>5.1f}°\n"
                    info += f"  Mín: {s['min']:>4.0f}° | Máx: {s['max']:>4.0f}°\n\n"
            t_dynamic.set_text(info)

            if i < self.proc.n_frames - 1:
                p_prox = self.proc.angulos_df.iloc[i+1]
                pares = [
                    ('Quad_D', 'Joel_D', ptr_qjd, txt_qj_d),
                    ('Joel_D', 'Torn_D', ptr_jtd, txt_jt_d),
                    ('Quad_E', 'Joel_E', ptr_qje, txt_qj_e),
                    ('Joel_E', 'Torn_E', ptr_jte, txt_jt_e)
                ]

                for j_prox, j_dist, ptr, txt in pares:
                    dx = p_prox[j_prox] - row[j_prox]
                    dy = p_prox[j_dist] - row[j_dist]
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
        st.sidebar.error("A planilha precisa ter as colunas 'ID' e 'ALTURA'. O sistema encontrou: " + ", ".join(df_antropo.columns))
        df_antropo = None

st.sidebar.markdown("---")
st.sidebar.markdown("### Sobre o Sistema")
st.sidebar.info("GPBIO: Análise Biomecânica de Marcha.")
st.sidebar.markdown("**Desenvolvido por Arthur Lins**")
	

if 'processadores' not in st.session_state:
    st.session_state.processadores = []

st.subheader("📁 Importação de Dados e Separação de Grupos")
st.info("Digite os nomes dos grupos do seu estudo e faça o upload dos arquivos .c3d dinâmicos em suas respectivas áreas.")

# Cria duas colunas visuais para o Upload
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
            except:
                pass 
                
            progress_bar.progress((i + 1) / len(arquivos_para_processar))
            
        st.success(f"✅ {len(st.session_state.processadores)} arquivos processados e agrupados com sucesso!")


if st.session_state.processadores:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Tabela de Médias", 
        "📈 Gráficos de Curvas", 
        "⚙️ Coordenação (Angle-Angle)", 
        "🎥 Animações 3D (GIFs)",
        "📦 Estatística (Boxplots e Barras)",
        "🧪 Testes de Hipótese",
        "📝 Relatório Clínico"
    ])

    # =========================================================================
    # TAB 1: TABELA DE MÉDIAS (DADOS BRUTOS + MÉTRICAS AVANÇADAS + SUMÁRIOS)
    # =========================================================================
    with tab1:
        st.subheader("📊 Tabela de Dados Brutos e Estatística Descritiva")
        st.write("Visão geral de todos os parâmetros espaço-temporais, cinemáticos e de coordenação, incluindo métricas avançadas (CAV e Transições). Ao final da tabela, são apresentadas as médias e desvios padrão de cada grupo.")
        
        dados_tabela = []
        for p in st.session_state.processadores:
            try:
                # 1. Parâmetros Espaço-Temporais Básicos
                linha = {
                    "Arquivo": p.nome_arq,
                    "Grupo": p.grupo,
                    "Velocidade (m/s)": p.velocidade_media if hasattr(p, 'velocidade_media') else np.nan,
                    "Apoio DIR (%)": p.fases_marcha.get('D', {}).get('Apoio', np.nan),
                    "Apoio ESQ (%)": p.fases_marcha.get('E', {}).get('Apoio', np.nan),
                    "Clearance DIR (mm)": p.foot_clearance.get('D', np.nan),
                    "Clearance ESQ (mm)": p.foot_clearance.get('E', np.nan),
                    "Passo DIR (mm)": p.comprimento_passo.get('D', np.nan),
                    "Passo ESQ (mm)": p.comprimento_passo.get('E', np.nan),
                    "Passo DIR (% Altura)": p.passo_norm.get('D', np.nan) if hasattr(p, 'passo_norm') else np.nan,
                    "Passo ESQ (% Altura)": p.passo_norm.get('E', np.nan) if hasattr(p, 'passo_norm') else np.nan,
                }
                
                # 2. Parâmetros Cinemáticos (Picos Máximos e Mínimos)
                stats = p.obter_stats()
                if stats:
                    for art in ['Quad_D', 'Quad_E', 'Joel_D', 'Joel_E', 'Torn_D', 'Torn_E']:
                        linha[f"{art} Máx (°)"] = stats.get(art, {}).get('max', np.nan)
                        linha[f"{art} Mín (°)"] = stats.get(art, {}).get('min', np.nan)

                # 3. Coordenação Vetorial (CAV, Transições e Frequências Fatiadas)
                pares_coord = [('QJ_DIR', 'Quad_Joel_D'), ('QJ_ESQ', 'Quad_Joel_E'),
                               ('JT_DIR', 'Joel_Torn_D'), ('JT_ESQ', 'Joel_Torn_E')]
                padroes = ['Proximal', 'EmFase', 'Distal', 'AntiFase']

                for par_label, par_key in pares_coord:
                    try:
                        prox_name, dist_name, lado = par_key.split('_')[0], par_key.split('_')[1], par_key.split('_')[2]
                        hss = p.eventos[lado]['HS']
                        
                        # --- CÁLCULO DO CAV E TRANSIÇÕES ---
                        if len(hss) > 1:
                            c_prox = p.extrair_ciclos_normalizados(p.angulos_df[f"{prox_name}_{lado}"].values, hss)
                            c_dist = p.extrair_ciclos_normalizados(p.angulos_df[f"{dist_name}_{lado}"].values, hss)
                            
                            if len(c_prox) > 0 and len(c_dist) > 0:
                                arr_p, arr_d = np.array(c_prox), np.array(c_dist)
                                delta_p, delta_d = np.diff(arr_p, axis=1), np.diff(arr_d, axis=1)
                                gamma_rad = np.arctan2(delta_d, delta_p)
                                gamma_deg = (np.degrees(gamma_rad) + 360) % 360
                                
                                # CAV (Variabilidade)
                                x_m, y_m = np.mean(np.cos(gamma_rad), axis=0), np.mean(np.sin(gamma_rad), axis=0)
                                r = np.clip(np.sqrt(x_m**2 + y_m**2), 0, 1)
                                cav_mean = np.mean(np.sqrt(2 * (1 - r)) * (180 / np.pi))
                                linha[f"CAV {par_label} (°)"] = cav_mean
                                
                                # Transições (Fluidez)
                                bins = [0, 45, 135, 225, 315, 360]
                                padroes_idx = np.digitize(gamma_deg, bins)
                                padroes_idx[padroes_idx == 5] = 1
                                mudancas = np.sum(np.diff(padroes_idx, axis=1) != 0, axis=1)
                                trans_mean = np.mean(mudancas)
                                linha[f"Transições {par_label}"] = trans_mean
                            else:
                                linha[f"CAV {par_label} (°)"] = np.nan
                                linha[f"Transições {par_label}"] = np.nan
                        else:
                            linha[f"CAV {par_label} (°)"] = np.nan
                            linha[f"Transições {par_label}"] = np.nan

                        # --- FREQUÊNCIAS FATIADAS (APOIO/BALANÇO) ---
                        if hasattr(p, 'coord_vetorial_series') and par_key in p.coord_vetorial_series:
                            serie = p.coord_vetorial_series[par_key]
                        else:
                            idxs = np.digitize(gamma_deg[0] if 'gamma_deg' in locals() else [], [0, 45, 135, 225, 315, 360])
                            map_labels = {1: 'Proximal', 2: 'EmFase', 3: 'Distal', 4: 'AntiFase', 5: 'Proximal'}
                            serie = [map_labels[i] for i in idxs] if len(idxs) > 0 else []

                        fatia_apoio = serie[0:60] if len(serie) >= 60 else []
                        fatia_balanco = serie[60:] if len(serie) > 60 else []

                        for padrao in padroes:
                            val_apoio = (fatia_apoio.count(padrao) / len(fatia_apoio)) * 100 if len(fatia_apoio) > 0 else np.nan
                            linha[f"APOIO {par_label} - {padrao} (%)"] = val_apoio
                            
                            val_balanco = (fatia_balanco.count(padrao) / len(fatia_balanco)) * 100 if len(fatia_balanco) > 0 else np.nan
                            linha[f"BALANÇO {par_label} - {padrao} (%)"] = val_balanco
                            
                    except Exception as e:
                        linha[f"CAV {par_label} (°)"] = np.nan
                        linha[f"Transições {par_label}"] = np.nan
                        pass
                
                # Arredonda tudo no final para deixar a tabela visualmente limpa
                for k, v in linha.items():
                    if isinstance(v, float) and not np.isnan(v):
                        linha[k] = round(v, 2)
                    elif isinstance(v, float) and np.isnan(v):
                        linha[k] = ""

                dados_tabela.append(linha)
            except Exception as e:
                continue
                
        if dados_tabela:
            df_tabela = pd.DataFrame(dados_tabela)
            
            # =================================================================
            # CÁLCULO NATIVO DE MÉDIA E DESVIO PADRÃO POR GRUPO
            # =================================================================
            summary_rows = []
            
            # Substitui strings vazias por NaN temporariamente para a matemática funcionar
            df_calc = df_tabela.replace("", np.nan)
            
            for grp in df_calc['Grupo'].unique():
                df_grp = df_calc[df_calc['Grupo'] == grp]
                
                mean_row = {"Arquivo": f"📌 MÉDIA - {grp}", "Grupo": grp}
                std_row = {"Arquivo": f"📉 DESVIO PADRÃO - {grp}", "Grupo": grp}
                
                for col in df_calc.columns:
                    if col not in ["Arquivo", "Grupo"]:
                        val_mean = df_grp[col].astype(float).mean()
                        val_std = df_grp[col].astype(float).std()
                        mean_row[col] = round(val_mean, 2) if pd.notnull(val_mean) else ""
                        std_row[col] = round(val_std, 2) if pd.notnull(val_std) else ""
                        
                summary_rows.append(mean_row)
                summary_rows.append(std_row)
            
            # Anexa as linhas de sumário no final da tabela
            df_summary = pd.DataFrame(summary_rows)
            df_final = pd.concat([df_tabela, df_summary], ignore_index=True)
            
            # Formatação visual para destacar as métricas no Streamlit
            def highlight_summary(row):
                if 'MÉDIA' in str(row['Arquivo']):
                    return ['font-weight: bold'] * len(row) # Azul claro
                elif 'DESVIO PADRÃO' in str(row['Arquivo']):
                    return ['font-weight: bold'] * len(row) # Vermelho claro
                return [''] * len(row)

            st.dataframe(df_final.style.apply(highlight_summary, axis=1), use_container_width=True, height=600)
            
            csv = df_final.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
            st.download_button(
                label="📥 Baixar Tabela de Dados com Médias e Desvios (CSV)",
                data=csv,
                file_name="dados_completos_com_estatistica.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.info("Importe e processe arquivos na barra lateral para gerar a tabela de resultados.")

    # =========================================================================
    # TAB 2: CURVAS CINEMÁTICAS
    # =========================================================================
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

                # Gráfico mais estreito para caber na metade da tela
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
                    if col == 0: ax.set_ylabel("Graus (°)", fontsize=9)
                    if row == 2: ax.set_xlabel("% Ciclo", fontsize=9)

                    if len(ciclos) > 0:
                        media, std = np.mean(ciclos, axis=0), np.std(ciclos, axis=0)
                        ax.fill_between(x_axis, media - std, media + std, color='gray', alpha=0.3)
                        ax.plot(x_axis, media, color='blue' if col==1 else 'red', lw=2)
                        ax.axhline(0, color='black', lw=0.8)
                    else:
                        ax.text(50, 0, "Sem Dados", ha='center')

                plt.tight_layout()
                st.pyplot(fig)

    # =========================================================================
    # TAB 3: COORDENAÇÃO VETORIAL AVANÇADA (O CENTRO DE CONTROLE)
    # =========================================================================
    with tab3:
        st.subheader("⚙️ Coordenação Vetorial e Controle Motor")
        
        # --- TEXTO EDUCATIVO E CLÍNICO ---
        with st.expander("📖 Dicionário Clínico: Entenda as Métricas de Coordenação", expanded=False):
            st.markdown("""
            O **Vector Coding** (Ângulo de Acoplamento) avalia **como** duas articulações trabalham juntas ao longo do tempo.
            
            **1. Frequência dos Padrões (Gráfico de Barras):**
            * <span style='color:#e74c3c'>**Dominância Proximal (Vermelho):**</span> Articulação superior guia o movimento (ex: Quadril move, Joelho estabiliza).
            * <span style='color:#2ecc71'>**Em Fase (Verde):**</span> Ambas fletem ou estendem juntas. Típico de movimentos fluidos e saudáveis.
            * <span style='color:#3498db'>**Dominância Distal (Azul):**</span> Articulação inferior guia o movimento.
            * <span style='color:#f1c40f'>**Anti-Fase (Amarelo):**</span> Uma flete enquanto a outra estende. Essencial para absorção de impacto, mas o excesso indica descoordenação patológica.
            
            **2. Métricas de Estabilidade (Boxplots):**
            * **Variabilidade (CAV):** Mede o "desvio padrão circular" da coordenação. Um sistema saudável tem adaptabilidade (CAV moderado). CAV muito alto = instabilidade motora (risco de queda). CAV muito baixo = rigidez articular e congelamento (comum em Parkinson rígido-acinético).
            * **Taxa de Transições:** Conta quantas vezes o paciente "troca de marcha" entre os 4 padrões em um único passo. Mede a fluidez do comando neuromuscular.
            """, unsafe_allow_html=True)

        st.markdown("---")
        grupos_estudo = sorted(list(set([p.grupo for p in st.session_state.processadores])))
        
        # Definição do layout de cores
        cor_g1 = '#a8c8f9'
        cor_g2 = '#f9a8a8'
        padroes = {'Proximal': '#e74c3c', 'EmFase': '#2ecc71', 'Distal': '#3498db', 'AntiFase': '#f1c40f'}

        # Dicionários para acomodar dados extraídos
        dados_coord = {g: { 'Quad_Joel_D': [], 'Quad_Joel_E': [], 'Joel_Torn_D': [], 'Joel_Torn_E': [] } for g in grupos_estudo}
        freq_acumulada = {g: { 'Quad-Joel (DIR)': {k: [] for k in padroes}, 'Quad-Joel (ESQ)': {k: [] for k in padroes},
                               'Joel-Torn (DIR)': {k: [] for k in padroes}, 'Joel-Torn (ESQ)': {k: [] for k in padroes} } for g in grupos_estudo}

        # Extração em massa e cálculo On-The-Fly
        cav_data = {'Quad_Joel': {g: [] for g in grupos_estudo}, 'Joel_Torn': {g: [] for g in grupos_estudo}}
        trans_data = {'Quad_Joel': {g: [] for g in grupos_estudo}, 'Joel_Torn': {g: [] for g in grupos_estudo}}

        for p in st.session_state.processadores:
            grp = p.grupo
            
            # Coleta de Frequências Globais (Para o Gráfico de Barras)
            map_coord = [('Quad_Joel_D', 'Quad-Joel (DIR)'), ('Quad_Joel_E', 'Quad-Joel (ESQ)'),
                         ('Joel_Torn_D', 'Joel-Torn (DIR)'), ('Joel_Torn_E', 'Joel-Torn (ESQ)')]
            for c_old, c_new in map_coord:
                freqs = p.coord_vetorial.get(c_old, {})
                for k in padroes.keys():
                    val = freqs.get(k, np.nan)
                    if not np.isnan(val): freq_acumulada[grp][c_new][k].append(val)

            # Coleta de Ciclos Brutos (Para os Diagramas de Fase e Cálculo de CAV/Transições)
            for prox, dist, label_cav in [('Quad', 'Joel', 'Quad_Joel'), ('Joel', 'Torn', 'Joel_Torn')]:
                for lado in ['D', 'E']:
                    chave_prox, chave_dist = f"{prox}_{lado}", f"{dist}_{lado}"
                    chave_par = f"{prox}_{dist}_{lado}"
                    
                    hss = p.eventos[lado]['HS']
                    if len(hss) > 1:
                        c_prox = p.extrair_ciclos_normalizados(p.angulos_df[chave_prox].values, hss)
                        c_dist = p.extrair_ciclos_normalizados(p.angulos_df[chave_dist].values, hss)
                        dados_coord[grp][chave_par].extend((c_prox, c_dist))
                        
                        # Motor Matemático para CAV e Transições
                        if len(c_prox) > 0 and len(c_dist) > 0:
                            arr_p, arr_d = np.array(c_prox), np.array(c_dist)
                            delta_p, delta_d = np.diff(arr_p, axis=1), np.diff(arr_d, axis=1)
                            
                            gamma_rad = np.arctan2(delta_d, delta_p)
                            gamma_deg = (np.degrees(gamma_rad) + 360) % 360
                            
                            # Transições
                            bins = [0, 45, 135, 225, 315, 360]
                            padroes_idx = np.digitize(gamma_deg, bins)
                            padroes_idx[padroes_idx == 5] = 1 # Ajuste do quadrante circular
                            mudancas = np.sum(np.diff(padroes_idx, axis=1) != 0, axis=1)
                            trans_data[label_cav][grp].append(np.mean(mudancas))
                            
                            # CAV (Variabilidade do Ângulo de Acoplamento)
                            x_m, y_m = np.mean(np.cos(gamma_rad), axis=0), np.mean(np.sin(gamma_rad), axis=0)
                            r = np.clip(np.sqrt(x_m**2 + y_m**2), 0, 1)
                            cav_mean = np.mean(np.sqrt(2 * (1 - r)) * (180 / np.pi))
                            cav_data[label_cav][grp].append(cav_mean)

        # -------------------------------------------------------------
        # PARTE 1: DIAGRAMAS DE FASE
        # -------------------------------------------------------------
        st.markdown("### 1. Comportamento Espacial (Diagramas Angle-Angle)")
	
	
        cols_t3 = st.columns(len(grupos_estudo))
        

        for idx, grp in enumerate(grupos_estudo):
            with cols_t3[idx]:
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

                fig_coord, axs_coord = plt.subplots(2, 2, figsize=(7, 7))
                pares_plot = [
                    (axs_coord[0, 0], 'D', 'Quad', 'Joel', 'Quad(°)', 'Joel(°)'),
                    (axs_coord[0, 1], 'E', 'Quad', 'Joel', 'Quad(°)', 'Joel(°)'),
                    (axs_coord[1, 0], 'D', 'Joel', 'Torn', 'Joel(°)', 'Torn(°)'),
                    (axs_coord[1, 1], 'E', 'Joel', 'Torn', 'Joel(°)', 'Torn(°)')
                ]

                for ax, lado, prox, dist, label_x, label_y in pares_plot:
                    chave_prox, chave_dist = f"{prox}_{lado}", f"{dist}_{lado}"
                    ciclos_prox, ciclos_dist = np.array(dados_grp[chave_prox]), np.array(dados_grp[chave_dist])
                    
                    if len(ciclos_prox) > 0 and len(ciclos_dist) > 0:
                        media_prox, media_dist = np.mean(ciclos_prox, axis=0), np.mean(ciclos_dist, axis=0)
                        ax.plot(media_prox, media_dist, color='blue' if lado=='E' else 'red', lw=2)
                        ax.scatter(media_prox[0], media_dist[0], color='green', s=60, zorder=5)
                        ax.scatter(media_prox[60], media_dist[60], color='orange', marker='X', s=60, zorder=5)
                        
                    ax.set_xlabel(label_x, fontsize=9); ax.set_ylabel(label_y, fontsize=9)
                    ax.set_title(f"{prox}-{dist} ({lado})", fontweight='bold', fontsize=10)
                    ax.grid(True, linestyle='--', alpha=0.5)

                plt.tight_layout()
                st.pyplot(fig_coord)

        # -------------------------------------------------------------
        # PARTE 2: FREQUÊNCIA DE PADRÕES (DIVIDIDO POR APOIO E BALANÇO)
        # -------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 2. Distribuição por Fases (Apoio vs. Balanço)")
        
        # Criamos abas internas para não poluir o visual
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
                        chave_par_interna = par.replace("-", "_").replace(" (DIR)", "_D").replace(" (ESQ)", "_E")
                        f_prox, f_fase, f_dist, f_anti = 0, 0, 0, 0
                        contagem = 0
                        
                        for p in [proc for proc in st.session_state.processadores if proc.grupo == grp]:
                            try:
                                if hasattr(p, 'coord_vetorial_series') and chave_par_interna in p.coord_vetorial_series:
                                    fatia = p.coord_vetorial_series[chave_par_interna][inicio:fim]
                                else:
                                    prox_name, dist_name = chave_par_interna.split('_')[0], chave_par_interna.split('_')[1]
                                    lado = chave_par_interna.split('_')[2]
                                    
                                    ciclo_p = np.mean(p.extrair_ciclos_normalizados(p.angulos_df[f"{prox_name}_{lado}"].values, p.eventos[lado]['HS']), axis=0)
                                    ciclo_d = np.mean(p.extrair_ciclos_normalizados(p.angulos_df[f"{dist_name}_{lado}"].values, p.eventos[lado]['HS']), axis=0)
                                    
                                    dp, dd = np.diff(ciclo_p), np.diff(ciclo_d)
                                    gamma = (np.degrees(np.arctan2(dd, dp)) + 360) % 360
                                    
                                    bins = [0, 45, 135, 225, 315, 360]
                                    idxs = np.digitize(gamma, bins)
                                    map_labels = {1: 'Proximal', 2: 'EmFase', 3: 'Distal', 4: 'AntiFase', 5: 'Proximal'}
                                    fatia = [map_labels[i] for i in idxs[inicio:fim]]
                                
                                total_fatia = len(fatia)
                                if total_fatia > 0:
                                    f_prox += fatia.count('Proximal') / total_fatia
                                    f_fase += fatia.count('EmFase') / total_fatia
                                    f_dist += fatia.count('Distal') / total_fatia
                                    f_anti += fatia.count('AntiFase') / total_fatia
                                    contagem += 1
                            except: continue
                        
                        denom = contagem if contagem > 0 else 1
                        m_prox.append((f_prox/denom)*100); m_fase.append((f_fase/denom)*100)
                        m_dist.append((f_dist/denom)*100); m_anti.append((f_anti/denom)*100)

                    x, width = np.arange(len(labels_pares)), 0.55
                    
                    # Salva as barras em variáveis para podermos colocar os textos dentro delas
                    bar1 = ax_bar.bar(x, m_prox, width, label='Proximal', color=padroes['Proximal'])
                    bar2 = ax_bar.bar(x, m_fase, width, bottom=m_prox, label='Em Fase', color=padroes['EmFase'])
                    bar3 = ax_bar.bar(x, m_dist, width, bottom=np.add(m_prox, m_fase), label='Distal', color=padroes['Distal'])
                    bar4 = ax_bar.bar(x, m_anti, width, bottom=np.add(np.add(m_prox, m_fase), m_dist), label='Anti-Fase', color=padroes['AntiFase'])

                    # --- ADIÇÃO 1: Rótulos de Porcentagem em Preto ---
                    for bar_group in [bar1, bar2, bar3, bar4]:
                        # A condição > 2.0 oculta o texto em faixas muito finas para não sobrepor números
                        lbls = [f"{v.get_height():.1f}%" if v.get_height() > 2.0 else "" for v in bar_group]
                        ax_bar.bar_label(bar_group, labels=lbls, label_type='center', color='black', fontweight='bold', fontsize=9)

                    ax_bar.set_ylabel('Frequência (%)', fontweight='bold')
                    ax_bar.set_xticks(x); ax_bar.set_xticklabels(labels_pares, fontweight='bold', rotation=15, fontsize=8)
                    ax_bar.set_ylim(0, 105)
                    
                    # Limpeza visual: remove linhas de borda superior e direita
                    ax_bar.spines['top'].set_visible(False)
                    ax_bar.spines['right'].set_visible(False)
                    
                    # --- ADIÇÃO 2: Legenda (Apenas no gráfico da extrema direita) ---
                    if idx == len(grupos_estudo) - 1:
                        leg = ax_bar.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title="Padrões")
                        leg.get_title().set_fontweight('bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig_bar)

        # Renderiza os gráficos nas sub-abas
        plot_fase_especifica(sub_tab_apoio, 0, 60, "Apoio")
        plot_fase_especifica(sub_tab_balanco, 60, 100, "Balanço")

                
        # -------------------------------------------------------------
        # PARTE 3: COMPLEXIDADE E ESTABILIDADE (COLUNAS DE MÉDIA + DP)
        # -------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 3. Índices de Estabilidade e Fluidez Motora")
        
        col_cav, col_trans = st.columns(2)
        
        def plot_grouped_bars(ax, dict_data, titulo, ylabel):
            labels_grupos = list(dict_data.keys())
            # Calcula Média e DP para cada grupo
            means = [np.mean(dict_data[g]) if dict_data[g] else 0 for g in labels_grupos]
            stds = [np.std(dict_data[g]) if dict_data[g] else 0 for g in labels_grupos]
            
            x_pos = np.arange(len(labels_grupos))
            # Cores consistentes: Azul para G1, Vermelho para G2
            cores = [cor_g1, cor_g2] if len(labels_grupos) > 1 else [cor_g1]
            
            bars = ax.bar(x_pos, means, yerr=stds, capsize=8, color=cores, edgecolor='black', alpha=0.9, width=0.6)
            
            # Adiciona os rótulos de valor no topo das barras
            ax.bar_label(bars, fmt='%.1f', padding=3, fontweight='bold')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels_grupos, fontweight='bold')
            ax.set_title(titulo, fontweight='bold', fontsize=12)
            ax.set_ylabel(ylabel)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        with col_cav:
            fig_cav, axs_cav = plt.subplots(1, 2, figsize=(10, 5))
            plot_grouped_bars(axs_cav[0], cav_data['Quad_Joel'], "Variabilidade (CAV)\nQuad-Joel", "Graus (°)")
            plot_grouped_bars(axs_cav[1], cav_data['Joel_Torn'], "Variabilidade (CAV)\nJoel-Torn", "Graus (°)")
            plt.tight_layout()
            st.pyplot(fig_cav)
            
        with col_trans:
            fig_tr, axs_tr = plt.subplots(1, 2, figsize=(10, 5))
            plot_grouped_bars(axs_tr[0], trans_data['Quad_Joel'], "Taxa de Transições\nQuad-Joel", "Mudanças / Ciclo")
            plot_grouped_bars(axs_tr[1], trans_data['Joel_Torn'], "Taxa de Transições\nJoel-Torn", "Mudanças / Ciclo")
            plt.tight_layout()
            st.pyplot(fig_tr)

    # =========================================================================
    # TAB 4: GERADOR DE GIFs
    # =========================================================================
    with tab4:
        st.subheader("Gerador de GIFs e Biofeedback Visual")
        st.info("Escolha os arquivos que deseja animar e a velocidade de reprodução.")
        lista_nomes = [p.nome_arq for p in st.session_state.processadores]
        col_selecao, col_vel = st.columns([2, 1])
        with col_selecao: selecionados = st.multiselect("Selecione os arquivos para animar:", lista_nomes)
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
                                st.download_button(f"📥 Baixar GIF ({p.nome_arq})", data=file_gif, file_name=f"{p.nome_arq.split('.')[0]}_3D.gif", mime="image/gif")
                        else: st.error(f"Falha: {msg}")

    # =========================================================================
    # TAB 5: ESTATÍSTICA (BOXPLOTS E COLUNAS EMPILHADAS)
    # =========================================================================
    with tab5:
        st.subheader("Análise Estatística Avançada (Comparação)")
        grupos_estudo = sorted(list(set([p.grupo for p in st.session_state.processadores])))
        
        # 1. Definição das Cores Exclusivas para os Grupos
        cor_g1 = '#a8c8f9' # Azul pastel (Grupo 1)
        cor_g2 = '#f9a8a8' # Vermelho pastel (Grupo 2)
        cores_grupos = {grupos_estudo[0]: cor_g1}
        if len(grupos_estudo) > 1:
            cores_grupos[grupos_estudo[1]] = cor_g2

        # 2. Dicionários que separam as variáveis
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

        # -------------------------------------------------------------
        # PLOT 1: BOXPLOTS e COLUNAS (IMAGENS SEPARADAS E ESPAÇOSAS)
        # -------------------------------------------------------------
        st.markdown("### 1. Parâmetros Espaço-Temporais")

        def gerar_colunas_passo_norm(dict_dados, titulo):
            fig, ax = plt.subplots(figsize=(8, 6))
            labels = list(dict_dados.keys())
            
            # Calcula Média e Desvio Padrão
            means = [np.nanmean(dict_dados[l]) if dict_dados[l] else 0 for l in labels]
            stds = [np.nanstd(dict_dados[l]) if dict_dados[l] else 0 for l in labels]
            
            # Define as cores baseadas no grupo (Azul para o primeiro, Vermelho para o segundo)
            cores = []
            for l in labels:
                if grupos_estudo[0] in l: cores.append(cor_g1)
                else: cores.append(cor_g2)

            bars = ax.bar(labels, means, yerr=stds, capsize=10, color=cores, edgecolor='black', alpha=0.8)
            
            # Adiciona o bloco de texto com Média e DP em Porcentagem
            for i, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 1, 
                        f"Média: {means[i]:.0f}%\nDP: ±{stds[i]:.1f}%", 
                        ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

            ax.set_title(titulo, fontweight='bold', fontsize=14)
            ax.set_ylabel("Porcentagem da Estatura (%)", fontsize=12)
            ax.set_ylim(0, 100) # Escala de porcentagem travada
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            return fig
        
        # Função para gerar um gráfico grande e isolado
        def gerar_boxplot_isolado(dict_dados, titulo, ylabel):
            fig, ax = plt.subplots(figsize=(8, 6)) # Gráfico grande e proporcional
            labels = list(dict_dados.keys())
            dados_limpos = [dict_dados[l] for l in labels if len(dict_dados[l]) > 0]
            labels_limpos = [l for l in labels if len(dict_dados[l]) > 0]
            
            if dados_limpos:
                bp = ax.boxplot(dados_limpos, patch_artist=True, labels=labels_limpos)
                
                # Aplica as cores correspondentes aos grupos (Azul ou Vermelho)
                for i, patch in enumerate(bp['boxes']): 
                    grp_name = labels_limpos[i].split('(')[0] if '(' in labels_limpos[i] else labels_limpos[i]
                    cor = cores_grupos.get(grp_name, '#dddddd')
                    patch.set_facecolor(cor)
                    
                for median in bp['medians']: median.set(color='black', linewidth=2)
                
                # Adiciona o texto Média/DP com bastante espaço (offset de 1.1)
                for i, d in enumerate(dados_limpos):
                    media, dp, mediana = np.mean(d), np.std(d), np.median(d)
                    ax.text(i + 1.10, mediana, f"M: {media:.1f}\nDP: {dp:.1f}", 
                            ha='left', va='center', fontsize=10, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'))
                
                # Aumenta o eixo X agressivamente para caber a caixa de texto
                ax.set_xlim(0.5, len(dados_limpos) + 0.9) 
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin - (ymax-ymin)*0.1, ymax + (ymax-ymin)*0.1)
                ax.set_title(titulo, fontweight='bold', fontsize=14)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            return fig

        # Montagem dos Dicionários
        dict_vel = {g: v_vel[g] for g in grupos_estudo}
        dict_ap = {}; dict_fc = {}; dict_ps = {}; dict_ps_norm = {}
        for g in grupos_estudo:
            dict_ap[f"{g}(D)"] = v_ap_d[g]; dict_ap[f"{g}(E)"] = v_ap_e[g]
            dict_fc[f"{g}(D)"] = v_fc_d[g]; dict_fc[f"{g}(E)"] = v_fc_e[g]
            dict_ps[f"{g}(D)"] = v_ps_d[g]; dict_ps[f"{g}(E)"] = v_ps_e[g]
            dict_ps_norm[f"{g}(D)"] = v_ps_norm_d[g]; dict_ps_norm[f"{g}(E)"] = v_ps_norm_e[g]
            
        # Exibição em Grid para melhor visualização
        col_box1, col_box2 = st.columns(2)
        with col_box1:
            st.pyplot(gerar_boxplot_isolado(dict_vel, "Velocidade Média", "m/s"))
            st.pyplot(gerar_boxplot_isolado(dict_fc, "Foot Clearance", "mm"))
        with col_box2:
            st.pyplot(gerar_boxplot_isolado(dict_ap, "Fase de Apoio", "% do Ciclo"))
            st.pyplot(gerar_boxplot_isolado(dict_ps, "Comprimento do Passo (Absoluto)", "mm"))
            
        # --- CRIANDO A LINHA DE BAIXO PARA O PASSO NORMALIZADO ---
        st.markdown("---")
        col_box3, col_box4 = st.columns(2) # <-- AQUI ESTÁ A CRIAÇÃO DA COLUNA 3!
        
        with col_box3:
            st.pyplot(gerar_colunas_passo_norm(dict_ps_norm, "Comprimento do Passo (% Altura)"))    


    # =========================================================================
    # TAB 6: ESTATÍSTICA INFERENCIAL (COM SEPARAÇÃO APOIO / BALANÇO)
    # =========================================================================
    with tab6:
        st.subheader("🧪 Testes de Hipótese e Significância (Completo)")
                # --- FLUXO DE PROCESSAMENTO ESTATÍSTICO ---
        with st.expander("🔍 Entenda o Fluxo de Decisão Estatística", expanded=True):
            st.markdown("""
            O sistema segue este protocolo para garantir a validade das comparações entre grupos:
            1.  **Teste de Normalidade (Shapiro-Wilk):** * Se *p > 0.05* em ambos os grupos: A distribuição é considerada **Normal**.
                * Se *p < 0.05* em qualquer grupo: A distribuição é **Não Normal**.
            2.  **Teste de Homocedasticidade (Levene):**
                * Avalia se as variâncias dos grupos são iguais (Homogêneas).
            3.  **Escolha do Teste de Hipótese:**
                * **Normal + Variâncias Iguais:** Teste T de Student (Paramétrico).
                * **Normal + Variâncias Diferentes:** Teste T de Welch (Paramétrico).
                * **Não Normal:** Teste de Mann-Whitney U (Não Paramétrico).
            """)

        grupos = sorted(list(set([p.grupo for p in st.session_state.processadores])))
        
        if len(grupos) < 2:
            st.warning("São necessários pelo menos 2 grupos diferentes para realizar testes comparativos.")
        else:
            st.info(f"Comparando: **{grupos[0]}** vs **{grupos[1]}**. Nível de significância: **α = 0.05**")
            
            # 1. Variáveis Base (Espaço-temporais e Cinemáticas)
            vars_estudo = [
                ("Velocidade (m/s)", 'attr', 'velocidade_media', None, None, None),
                ("Apoio DIR (%)", 'fases', 'D', 'Apoio', None, None),
                ("Apoio ESQ (%)", 'fases', 'E', 'Apoio', None, None),
                ("Clearance DIR (mm)", 'clearance', 'D', None, None, None),
                ("Clearance ESQ (mm)", 'clearance', 'E', None, None, None),
                ("Passo DIR (mm)", 'passo', 'D', None, None, None),
                ("Passo ESQ (mm)", 'passo', 'E', None, None, None),
                ("Passo DIR Norm (% Altura)", 'passo_norm', 'D', None, None, None),
                ("Passo ESQ Norm (% Altura)", 'passo_norm', 'E', None, None, None),
                ("Quad DIR: Máx (°)", 'stats', 'Quad_D', 'max', None, None),
                ("Quad DIR: Mín (°)", 'stats', 'Quad_D', 'min', None, None),
                ("Quad ESQ: Máx (°)", 'stats', 'Quad_E', 'max', None, None),
                ("Quad ESQ: Mín (°)", 'stats', 'Quad_E', 'min', None, None),
                ("Joel DIR: Máx (°)", 'stats', 'Joel_D', 'max', None, None),
                ("Joel DIR: Mín (°)", 'stats', 'Joel_D', 'min', None, None),
                ("Joel ESQ: Máx (°)", 'stats', 'Joel_E', 'max', None, None),
                ("Joel ESQ: Mín (°)", 'stats', 'Joel_E', 'min', None, None),
                ("Torn DIR: Máx (°)", 'stats', 'Torn_D', 'max', None, None),
                ("Torn DIR: Mín (°)", 'stats', 'Torn_D', 'min', None, None),
                ("Torn ESQ: Máx (°)", 'stats', 'Torn_E', 'max', None, None),
                ("Torn ESQ: Mín (°)", 'stats', 'Torn_E', 'min', None, None),
            ]

            # 2. Geração Dinâmica das Variáveis de Coordenação por FASE
            pares_coord = [('QJ DIR', 'Quad_Joel_D'), ('QJ ESQ', 'Quad_Joel_E'), 
                           ('JT DIR', 'Joel_Torn_D'), ('JT ESQ', 'Joel_Torn_E')]
            padroes_coord = ['Proximal', 'EmFase', 'Distal', 'AntiFase']
            
            for par_label, par_key in pares_coord:
                for padrao in padroes_coord:
                    # Fase de Apoio (0 a 60)
                    vars_estudo.append((f"APOIO {par_label}: {padrao} (%)", 'coord_fase', par_key, padrao, 0, 60))
                    # Fase de Balanço (60 a 101)
                    vars_estudo.append((f"BALANÇO {par_label}: {padrao} (%)", 'coord_fase', par_key, padrao, 60, 101))

            results_table = []

            for label, cat, key1, key2, inicio, fim in vars_estudo:
                data_g1, data_g2 = [], []
                
                for p in st.session_state.processadores:
                    try:
                        val = np.nan
                        if cat == 'attr': val = getattr(p, key1)
                        elif cat == 'fases': val = p.fases_marcha.get(key1, {}).get(key2, np.nan)
                        elif cat == 'clearance': val = p.foot_clearance.get(key1, np.nan)
                        elif cat == 'passo': val = p.comprimento_passo.get(key1, np.nan)
                        elif cat == 'passo_norm': val = getattr(p, 'passo_norm', {}).get(key1, np.nan)
                        elif cat == 'stats':
                            s = p.obter_stats() or {}
                            val = s.get(key1, {}).get(key2, np.nan)
                        elif cat == 'coord_fase':
                            
                            if hasattr(p, 'coord_vetorial_series') and key1 in p.coord_vetorial_series:
                                fatia = p.coord_vetorial_series[key1][inicio:fim]
                            else:
                                prox_name, dist_name, lado = key1.split('_')[0], key1.split('_')[1], key1.split('_')[2]
                                ciclo_p = np.mean(p.extrair_ciclos_normalizados(p.angulos_df[f"{prox_name}_{lado}"].values, p.eventos[lado]['HS']), axis=0)
                                ciclo_d = np.mean(p.extrair_ciclos_normalizados(p.angulos_df[f"{dist_name}_{lado}"].values, p.eventos[lado]['HS']), axis=0)
                                dp, dd = np.diff(ciclo_p), np.diff(ciclo_d)
                                gamma = (np.degrees(np.arctan2(dd, dp)) + 360) % 360
                                idxs = np.digitize(gamma, [0, 45, 135, 225, 315, 360])
                                map_labels = {1: 'Proximal', 2: 'EmFase', 3: 'Distal', 4: 'AntiFase', 5: 'Proximal'}
                                fatia = [map_labels[i] for i in idxs[inicio:fim]]
                            
                            if len(fatia) > 0:
                                val = (fatia.count(key2) / len(fatia)) * 100
                        
                        if not np.isnan(val):
                            if p.grupo == grupos[0]: data_g1.append(val)
                            elif p.grupo == grupos[1]: data_g2.append(val)
                    except: continue

                        if is_normal:
                    if equal_var:
                        test_name = "Teste T (Normalidade e Homocedasticidade)"
                    else:
                        test_name = "Teste T - Welch (Normalidade e Heterocedasticidade)"
                    
                    _, p_val = sp_stats.ttest_ind(data_g1, data_g2, equal_var=equal_var)
                    
                    # Cálculo do Tamanho do Efeito (Cohen's d)
                    s_pooled = np.sqrt((np.var(data_g1, ddof=1) + np.var(data_g2, ddof=1)) / 2)
                    effect_size = (np.mean(data_g1) - np.mean(data_g2)) / s_pooled if s_pooled > 0 else 0
                else:
                    test_name = "Mann-Whitney (Não Paramétrico - Distribuição Não Normal)"
                    u_stat, p_val = sp_stats.mannwhitneyu(data_g1, data_g2, alternative='two-sided')
                    
                    # Cálculo do Tamanho do Efeito (r de Rosenthal)
                    effect_size = 1 - (2 * u_stat) / (len(data_g1) * len(data_g2))

                    })

            if results_table:
                res_df = pd.DataFrame(results_table)
                def highlight_sig(val):
                    return 'background-color: #d4edda;' if 'SIGNIFICANTE' in str(val) else ''

                st.dataframe(res_df.style.map(highlight_sig, subset=['Resultado']), use_container_width=True, height=600)
                csv_stat = res_df.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
                st.download_button("📥 Baixar Tabela Estatística para SPSS (CSV)", data=csv_stat, file_name="estatistica_faseada.csv", mime="text/csv")
		# ==========================================
                # EXPORTAÇÃO DA MATRIZ BRUTA PARA O SPSS
                # ==========================================
                st.markdown("---")
                st.markdown("#### 📤 Exportação para SPSS (Matriz de Dados Brutos)")
                st.info("Matriz bruta de dados para SPSS.")
                
                matriz_spss = []
                for p in st.session_state.processadores:
                    linha = {
                        "Arquivo": p.nome_arq,
                        "Grupo": p.grupo
                    }
                    
                    
                    for label, cat, key1, key2, inicio, fim in vars_estudo:
                        try:
                            val = np.nan
                            if cat == 'attr': val = getattr(p, key1)
                            elif cat == 'fases': val = p.fases_marcha.get(key1, {}).get(key2, np.nan)
                            elif cat == 'clearance': val = p.foot_clearance.get(key1, np.nan)
                            elif cat == 'passo': val = p.comprimento_passo.get(key1, np.nan)
                            elif cat == 'passo_norm': val = getattr(p, 'passo_norm', {}).get(key1, np.nan)
                            elif cat == 'stats': val = p.obter_stats().get(key1, {}).get(key2, np.nan)
                            elif cat == 'coord_fase':
                                if hasattr(p, 'coord_vetorial_series') and key1 in p.coord_vetorial_series:
                                    fatia = p.coord_vetorial_series[key1][inicio:fim]
                                    if len(fatia) > 0: val = (fatia.count(key2) / len(fatia)) * 100
                            
                            linha[label] = round(val, 4) if not np.isnan(val) else ""
                        except:
                            linha[label] = ""
                            
                    matriz_spss.append(linha)
                
                df_spss = pd.DataFrame(matriz_spss)
                csv_spss = df_spss.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
                
                st.download_button(
                    label="📊 Baixar Matriz Bruta de Pacientes (Pronta para SPSS)", 
                    data=csv_spss, 
                    file_name="matriz_spss_gpbio.csv",
                    mime="text/csv",
                    type="primary"
                )
		
		# ==========================================
                # CÁLCULO NATIVO DA MANOVA 
                # ==========================================
                st.markdown("---")
                st.markdown("### 🧠 MANOVA Nativa (Statsmodels)")
                st.info("Cálculo multivariado independente. Evita a necessidade de exportação para softwares externos.")
                
                df_manova = pd.DataFrame(matriz_spss)
                
                
                import re
                df_manova.columns = [re.sub(r'\W+', '_', col).strip('_') for col in df_manova.columns]
                
                
                pares_teste = ['QJ_DIR', 'QJ_ESQ', 'JT_DIR', 'JT_ESQ']

                if st.button("🚀 Rodar MANOVA (Por Par Articular)", type="primary", use_container_width=True):
                    try:
                        # --- FASE DE APOIO ---
                        st.markdown("#### 🦵 Resultados Multivariados: Fase de Apoio")
                        for par in pares_teste:
                           
                            vars_par = [c for c in df_manova.columns if 'APOIO' in c and par in c and 'AntiFase' not in c]
                            
                            if len(vars_par) > 0:
                                formula = f"{' + '.join(vars_par)} ~ Grupo"
                                manova_res = MANOVA.from_formula(formula, data=df_manova).mv_test()
                                st.markdown(f"**Teste Específico: {par.replace('_', ' ')}**")
                                st.text(manova_res.summary())
                        
                        st.markdown("---")
                        
                        # --- FASE DE BALANÇO ---
                        st.markdown("#### ✈️ Resultados Multivariados: Fase de Balanço")
                        for par in pares_teste:
                            vars_par = [c for c in df_manova.columns if ('BALANÇO' in c or 'BALANCO' in c) and par in c and 'AntiFase' not in c]
                            
                            if len(vars_par) > 0:
                                formula = f"{' + '.join(vars_par)} ~ Grupo"
                                manova_res = MANOVA.from_formula(formula, data=df_manova).mv_test()
                                st.markdown(f"**Teste Específico: {par.replace('_', ' ')}**")
                                st.text(manova_res.summary())
                                
                        st.success("Cálculos concluídos! Modelos rodados de forma segmentada para proteger contra o erro de Matriz Singular em amostras pequenas.")
                        
                    except Exception as e:
                        st.error("Erro matemático. Verifique se importou arquivos suficientes ou se os grupos estão corretos.")
                        st.code(str(e))

    # =========================================================================
    # TAB 7: RELATÓRIO CLÍNICO AUTOMATIZADO (COM FASES)
    # =========================================================================
    with tab7:
        st.subheader("📝 Relatório Clínico e Achados Significativos")
        
        grupos = sorted(list(set([p.grupo for p in st.session_state.processadores])))
        
        if len(grupos) < 2:
            st.info("O relatório comparativo requer a importação de pelo menos dois grupos distintos.")
        else:
            g_controle = grupos[0]
            g_teste = grupos[1]
            
            st.markdown(f"Análise comparativa gerada entre o grupo base (**{g_controle}**) e o grupo de estudo (**{g_teste}**).")
            st.markdown("---")
            
            achados_espaco_temp = []
            achados_cinematica = []
            achados_coord_apoio = []
            achados_coord_balanco = []
            
            # 1. Mapeamento Base
            vars_relatorio = [
                ("Velocidade (m/s)", 'attr', 'velocidade_media', None, None, None, achados_espaco_temp),
                ("Apoio DIR (%)", 'fases', 'D', 'Apoio', None, None, achados_espaco_temp),
                ("Apoio ESQ (%)", 'fases', 'E', 'Apoio', None, None, achados_espaco_temp),
                ("Clearance DIR (mm)", 'clearance', 'D', None, None, None, achados_espaco_temp),
                ("Clearance ESQ (mm)", 'clearance', 'E', None, None, None, achados_espaco_temp),
                ("Passo DIR (mm)", 'passo', 'D', None, None, None, achados_espaco_temp),
                ("Passo ESQ (mm)", 'passo', 'E', None, None, None, achados_espaco_temp),
                ("Passo DIR Norm (% Altura)", 'passo_norm', 'D', None, None, None, achados_espaco_temp),
                ("Passo ESQ Norm (% Altura)", 'passo_norm', 'E', None, None, None, achados_espaco_temp),
                ("Quad DIR: Máx (°)", 'stats', 'Quad_D', 'max', None, None, achados_cinematica),
                ("Quad DIR: Mín (°)", 'stats', 'Quad_D', 'min', None, None, achados_cinematica),
                ("Quad ESQ: Máx (°)", 'stats', 'Quad_E', 'max', None, None, achados_cinematica),
                ("Quad ESQ: Mín (°)", 'stats', 'Quad_E', 'min', None, None, achados_cinematica),
                ("Joel DIR: Máx (°)", 'stats', 'Joel_D', 'max', None, None, achados_cinematica),
                ("Joel DIR: Mín (°)", 'stats', 'Joel_D', 'min', None, None, achados_cinematica),
                ("Joel ESQ: Máx (°)", 'stats', 'Joel_E', 'max', None, None, achados_cinematica),
                ("Joel ESQ: Mín (°)", 'stats', 'Joel_E', 'min', None, None, achados_cinematica),
                ("Torn DIR: Máx (°)", 'stats', 'Torn_D', 'max', None, None, achados_cinematica),
                ("Torn DIR: Mín (°)", 'stats', 'Torn_D', 'min', None, None, achados_cinematica),
                ("Torn ESQ: Máx (°)", 'stats', 'Torn_E', 'max', None, None, achados_cinematica),
                ("Torn ESQ: Mín (°)", 'stats', 'Torn_E', 'min', None, None, achados_cinematica),
            ]

            # 2. Geração Dinâmica para o Relatório Clínico
            pares_coord = [('QJ DIR', 'Quad_Joel_D'), ('QJ ESQ', 'Quad_Joel_E'), 
                           ('JT DIR', 'Joel_Torn_D'), ('JT ESQ', 'Joel_Torn_E')]
            
            for par_label, par_key in pares_coord:
                for padrao in ['Proximal', 'EmFase', 'Distal', 'AntiFase']:
                    vars_relatorio.append((f"{par_label} - {padrao}", 'coord_fase', par_key, padrao, 0, 60, achados_coord_apoio))
                    vars_relatorio.append((f"{par_label} - {padrao}", 'coord_fase', par_key, padrao, 60, 101, achados_coord_balanco))

            for label, cat, key1, key2, inicio, fim, lista_destino in vars_relatorio:
                data_g1, data_g2 = [], []
                for p in st.session_state.processadores:
                    try:
                        val = np.nan
                        if cat == 'attr': val = getattr(p, key1)
                        elif cat == 'fases': val = p.fases_marcha.get(key1, {}).get(key2, np.nan)
                        elif cat == 'clearance': val = p.foot_clearance.get(key1, np.nan)
                        elif cat == 'passo': val = p.comprimento_passo.get(key1, np.nan)
                        elif cat == 'passo_norm': val = getattr(p, 'passo_norm', {}).get(key1, np.nan)
                        elif cat == 'stats': val = p.obter_stats().get(key1, {}).get(key2, np.nan)
                        elif cat == 'coord_fase':
                            if hasattr(p, 'coord_vetorial_series') and key1 in p.coord_vetorial_series:
                                fatia = p.coord_vetorial_series[key1][inicio:fim]
                            else:
                                prox_name, dist_name, lado = key1.split('_')[0], key1.split('_')[1], key1.split('_')[2]
                                ciclo_p = np.mean(p.extrair_ciclos_normalizados(p.angulos_df[f"{prox_name}_{lado}"].values, p.eventos[lado]['HS']), axis=0)
                                ciclo_d = np.mean(p.extrair_ciclos_normalizados(p.angulos_df[f"{dist_name}_{lado}"].values, p.eventos[lado]['HS']), axis=0)
                                dp, dd = np.diff(ciclo_p), np.diff(ciclo_d)
                                idxs = np.digitize((np.degrees(np.arctan2(dd, dp)) + 360) % 360, [0, 45, 135, 225, 315, 360])
                                map_labels = {1: 'Proximal', 2: 'EmFase', 3: 'Distal', 4: 'AntiFase', 5: 'Proximal'}
                                fatia = [map_labels[i] for i in idxs[inicio:fim]]
                            
                            if len(fatia) > 0: val = (fatia.count(key2) / len(fatia)) * 100
                        
                        if not np.isnan(val):
                            if p.grupo == g_controle: data_g1.append(val)
                            elif p.grupo == g_teste: data_g2.append(val)
                    except: continue

                if len(data_g1) > 2 and len(data_g2) > 2:
                    _, p_norm1 = sp_stats.shapiro(data_g1)
                    _, p_norm2 = sp_stats.shapiro(data_g2)
                    is_normal = (p_norm1 > 0.05 and p_norm2 > 0.05)
                    
                    if is_normal:
                        _, p_val = sp_stats.ttest_ind(data_g1, data_g2, equal_var=(sp_stats.levene(data_g1, data_g2)[1] > 0.05))
                    else:
                        _, p_val = sp_stats.mannwhitneyu(data_g1, data_g2, alternative='two-sided')

                    if p_val < 0.05:
                        media_c, media_t = np.mean(data_g1), np.mean(data_g2)
                        if media_c > media_t:
                            texto = f"📉 **{label}**: Redução significativa no grupo {g_teste} (*p={p_val:.3f}*). O grupo **{g_controle}** apresentou média maior ({media_c:.1f} vs {media_t:.1f})."
                        else:
                            texto = f"📈 **{label}**: Aumento significativo no grupo {g_teste} (*p={p_val:.3f}*). O grupo **{g_teste}** superou a base de normalidade ({media_t:.1f} vs {media_c:.1f})."
                        lista_destino.append(texto)

            # Exibição do Relatório
            st.markdown("### 🚶 Parâmetros Espaço-Temporais")
            if achados_espaco_temp:
                for a in achados_espaco_temp: st.markdown(a)
            else: st.write("Nenhuma diferença significativa.")

            st.markdown("### 📐 Cinemática Articular (Amplitudes)")
            if achados_cinematica:
                for a in achados_cinematica: st.markdown(a)
            else: st.write("Nenhuma diferença significativa.")

            st.markdown("### 🦵 Coordenação na Fase de Apoio (0-60%)")
            st.info("Momento de aceitação de carga e estabilização postural no solo.")
            if achados_coord_apoio:
                for a in achados_coord_apoio: st.markdown(a)
            else: st.write("Nenhuma diferença significativa na coordenação de apoio.")

            st.markdown("### ✈️ Coordenação na Fase de Balanço (60-100%)")
            st.info("Momento de oscilação do membro e progressão da passada (risco de tropeço).")
            if achados_coord_balanco:
                for a in achados_coord_balanco: st.markdown(a)
            else: st.write("Nenhuma diferença significativa na coordenação de balanço.") 
