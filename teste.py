# Executando a simulação e gerando gráficos — saída visível ao usuário
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Recriar as funções e dados (copiado do seu código fornecido)
df_press_abs = pd.DataFrame({
    'Pressão Absoluta (bar)': [
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
        0.09, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60,
        0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40,
        1.50, 1.60, 1.80, 2.00, 2.20, 2.40, 2.60, 2.80, 3.00
    ],
    'Temperatura (°C)': [
        6.7, 17.2, 23.8, 28.6, 32.5, 35.8, 38.9, 41.2,
        43.9, 45.4, 53.6, 59.7, 68.7, 75.9, 81.3, 86.0,
        90.0, 93.5, 96.6, 99.6, 101.8, 104.8, 107.5, 109.9,
        112.0, 114.0, 117.6, 120.9, 124.0, 126.8, 129.5, 131.9, 134.2
    ],
    'Entalpia do Vapor (kcal/kg)': [
        600.1, 604.8, 607.7, 609.8, 611.5, 612.9, 613.9, 615.1,
        616.1, 617.0, 620.5, 623.1, 626.8, 629.4, 631.3, 632.9,
        634.2, 635.3, 636.3, 637.2, 638.0, 638.7, 639.4, 640.0,
        640.6, 641.1, 642.1, 643.0, 643.8, 644.5, 645.1, 645.7, 646.2
    ],
    'Calor Latente (kcal/kg)': [
        593.0, 587.4, 583.9, 581.1, 578.9, 577.1, 575.0, 574.1,
        572.9, 571.6, 567.0, 563.5, 558.2, 553.7, 550.2, 547.0,
        544.1, 541.6, 539.3, 537.1, 535.4, 533.1, 531.0, 529.1,
        527.3, 525.6, 522.4, 519.5, 516.7, 514.1, 511.7, 509.4, 507.2
    ]
})

def arrumar_lista(lista):
    return np.round(lista, 2).tolist()

def calcular_pressao_temp(seq, press_inicial, key_inicial, key_final, P_ref, T_ref, H_ref, L_ref):
    frac_queda = (key_inicial - key_final) / (seq - 1)
    queda_rel = [0] + (key_inicial - np.arange(seq) * frac_queda) / (10 * seq)
    queda_total = round((press_inicial - queda_rel[-1]) * 1.01972, 2)
    queda_entre = np.array(queda_rel) * queda_total

    press_efeitos = np.zeros(seq + 1)
    press_efeitos[0] = press_inicial
    for i in range(seq):
        press_efeitos[i + 1] = press_efeitos[i] - queda_entre[i]

    df = pd.DataFrame({
        'Efeito': np.arange(1, seq + 2),
        'Pressão (bar)': np.round(press_efeitos, 3),
        'Temperatura (°C)': np.round(np.interp(press_efeitos, P_ref, T_ref), 2),
        'Entalpia (kcal/kg)': np.round(np.interp(press_efeitos, P_ref, H_ref), 2),
        'Calor Latente (kcal/kg)': np.round(np.interp(press_efeitos, P_ref, L_ref), 2)
    })
    return df, queda_rel, queda_total, queda_entre

# função principal (idêntica à sua, mas retornando arrays diretamente)
def calcular_evaporadores_otimizado(
    df_press_abs=df_press_abs,
    brix_inicial: float = 14,
    vaz_caldo: float = 250,
    temp_inicial: float = 98,
    press_vapor: float = 2.0,
    perda_rad_frac: float = 0.005,
    perda_incond_frac: float = 0.015,
    listaEvap = [3500,3000,2500,2000,2000],
    alvo_brix_final: tuple = (60, 62)
) -> Dict[str, Any]:
    
    P_ref = df_press_abs["Pressão Absoluta (bar)"].values
    T_ref = df_press_abs["Temperatura (°C)"].values
    H_ref = df_press_abs["Entalpia do Vapor (kcal/kg)"].values
    L_ref = df_press_abs["Calor Latente (kcal/kg)"].values

    def simular(mul_1, mul_2, press_vapor_local):
        df_seq_impar, *_ = calcular_pressao_temp(3, press_vapor_local, 11.0, 9.0, P_ref, T_ref, H_ref, L_ref)
        df_seq_par, *_ = calcular_pressao_temp(2, press_vapor_local, 11.0, 9.0, P_ref, T_ref, H_ref, L_ref)

        pressao_98C = np.interp([temp_inicial], T_ref, P_ref)[0]
        entalpia_98C = np.interp([temp_inicial], T_ref, H_ref)[0]
        latente_98C = np.interp([temp_inicial], T_ref, L_ref)[0]

        Pressao_list = [press_vapor_local, press_vapor_local,
                        float(df_seq_impar.loc[1, "Pressão (bar)"]),
                        float(df_seq_par.loc[1, "Pressão (bar)"]),
                        float(df_seq_impar.loc[2, "Pressão (bar)"])]

        Entalpia_list = [entalpia_98C,
                         float(df_seq_impar.loc[0, "Entalpia (kcal/kg)"]),
                         float(df_seq_impar.loc[0, "Entalpia (kcal/kg)"]),
                         float(df_seq_impar.loc[1, "Entalpia (kcal/kg)"]),
                         float(df_seq_par.loc[1, "Entalpia (kcal/kg)"]),
                         float(df_seq_impar.loc[2, "Entalpia (kcal/kg)"])]

        Latente_list = [latente_98C,
                        float(df_seq_impar.loc[0, "Calor Latente (kcal/kg)"]),
                        float(df_seq_impar.loc[0, "Calor Latente (kcal/kg)"]),
                        float(df_seq_impar.loc[1, "Calor Latente (kcal/kg)"]),
                        float(df_seq_par.loc[1, "Calor Latente (kcal/kg)"]),
                        float(df_seq_impar.loc[2, "Calor Latente (kcal/kg)"])]

        quantidade = 5
        brix_teorico = np.linspace(brix_inicial, 68, quantidade + 1)
        brix_med = (brix_teorico[:-1] + brix_teorico[1:]) / 2
        EPE = np.concatenate(([0], (2 * brix_inicial) / (100 - brix_med)))

        temp_vapor = np.interp(Pressao_list, P_ref, T_ref)
        temp_caldo = [temp_inicial] + arrumar_lista(temp_vapor)
        temp_caldo_ajustada = [t + e for t, e in zip(temp_caldo, EPE)]

        vazao_list = np.zeros(quantidade + 1)
        brix_list = np.zeros(quantidade + 1)
        vazVap_list = np.zeros(quantidade + 1)
        Cp_list = np.zeros(quantidade + 1)
        ConsVap_list = np.zeros(quantidade)
        VapUtil_list = np.zeros(quantidade)
        vapGeradoTotal_list = np.zeros(quantidade)
        vazSangria_list = np.zeros(quantidade)
        taxaEvap = np.zeros(quantidade)

        vazao_list[0] = vaz_caldo * 1000
        brix_list[0] = brix_inicial
        Cp_list[0] = 1 - 0.006 * brix_list[0]

        for i in range(5):
            if i <= 1:
                Consumo_Vapor = vazao_list[i] * Cp_list[i] * (
                    (temp_caldo_ajustada[i + 1] - temp_caldo_ajustada[i]) / Entalpia_list[i]
                )
                ConsVap_list[i] = Consumo_Vapor
            else:
                ConsVap_list[i] = 0.0

            if i <= 1:
                VazVapor = Consumo_Vapor * (mul_1 if i == 0 else mul_2)
                vazVap_list[i] = VazVapor

            PerdRad = perda_rad_frac * vazVap_list[i]
            PerdIncond = perda_incond_frac * vazVap_list[i]
            VapUtil = vazVap_list[i] - PerdRad - PerdIncond - ConsVap_list[i]
            VapUtil_list[i] = VapUtil

            vapGerado = (Latente_list[i] / Latente_list[i + 1]) * VapUtil
            VazFlash = 0.0 if i <= 1 else vazao_list[i] * Cp_list[i] * (
                (temp_caldo_ajustada[i] - temp_caldo_ajustada[i + 1]) / Latente_list[i]
            )
            vapGeradoTotal = vapGerado + VazFlash
            vapGeradoTotal_list[i] = vapGeradoTotal

            if i == 0:
                keySangria = 100 / 350
            elif i <= 2:
                keySangria = 14 / 350
            else:
                keySangria = 0
            vazSangria_list[i] = vazao_list[i] * keySangria

            if i <= 2:
                vazVap_list[i + 2] = vapGeradoTotal_list[i] - vazSangria_list[i]

            vazao_list[i + 1] = vazao_list[i] - vapGeradoTotal_list[i]
            vazao_list[i + 1] = vazao_list[i + 1] if vazao_list[i + 1] != 0 else 1e-9
            brix_list[i + 1] = vazao_list[i] * brix_list[i] / vazao_list[i + 1]
            Cp_list[i + 1] = 1 - 0.006 * brix_list[i + 1]
            taxaEvap[i] = vapGeradoTotal_list[i]/listaEvap[i]

        return brix_list[-1], brix_list, ConsVap_list, vazVap_list, vapGeradoTotal, vazao_list, Cp_list, VapUtil_list, taxaEvap

    # Ajuste automático como antes, mas adaptado para usar press_vapor local
    melhor_brix, melhor_par, melhor_cons = None, (0, 0), 0.0
    for m1 in np.linspace(5, 15, 20):
        for m2 in np.linspace(1500, 2500, 20):
            brix_final, brix_list, ConsVap_list, vazVap_list, vapGeradoTotal, vazao_list, Cp_list, VapUtil_list, taxaEvap = simular(m1, m2, press_vapor)
            if alvo_brix_final[0] <= brix_final <= alvo_brix_final[1]:
                melhor_brix = brix_final
                melhor_par = (m1, m2)
                melhor_cons = float(np.sum(ConsVap_list))
                break
        if melhor_brix is not None:
            break

    if melhor_brix is None:
        alvo = 0.5 * (alvo_brix_final[0] + alvo_brix_final[1])
        melhor_erro = 1e12
        for m1 in np.linspace(5, 15, 40):
            for m2 in np.linspace(1200, 2800, 40):
                brix_final, brix_list, ConsVap_list, vazVap_list, vapGeradoTotal, vazao_list, Cp_list, VapUtil_list, taxaEvap = simular(m1, m2, press_vapor)
                erro = abs(brix_final - alvo)
                if erro < melhor_erro:
                    melhor_erro = erro
                    melhor_par = (m1, m2)
                    melhor_brix = brix_final
        brix_final, brix_lista, ConsVap_list, vazVap_list, vapGeradoTotal, vazao_list, Cp_list, VapUtil_list, taxaEvap = simular(*melhor_par, press_vapor)
        melhor_cons = float(np.sum(ConsVap_list))
    else:
        brix_final, brix_lista, ConsVap_list, vazVap_list, vapGeradoTotal, vazao_list, Cp_list, VapUtil_list, taxaEvap = simular(*melhor_par, press_vapor)

    consumo_total_vapor = float(np.sum(ConsVap_list))
    vapor_entrada_12 = float(vazVap_list[0] + vazVap_list[1])

    return {
        "Melhores Multiplicadores": melhor_par,
        "Brix Final": round(brix_final, 4),
        "Brix Efeitos": arrumar_lista(brix_lista),
        "Consumo Total de Vapor (kg/h)": round(consumo_total_vapor, 6),
        "Lista Consumo por Efeito (kg/h)": arrumar_lista(ConsVap_list),
        "Lista Vapor Entrada por Efeito (kg/h)": arrumar_lista(vazVap_list),
        "Vapor entrada efe1+efe2 (kg/h)": round(vapor_entrada_12, 2),
        "Taxa de Evaporação": taxaEvap
    }

# Agora: varrer pressões do vapor vivo e coletar consumo e brix final
pressures = np.linspace(1.5, 3.0, 15)  # 16 pontos entre 1.5 e 3.0 bar
results = []

for p in pressures:
    res = calcular_evaporadores_otimizado(press_vapor=float(p))
    results.append({
        "press_vapor": float(p),
        "consumo_total_vapor": res["Consumo Total de Vapor (kg/h)"],
        "vapor entrada": res["Vapor entrada efe1+efe2 (kg/h)"],
        "brix_final": res["Brix Final"],
        "mul_1": res["Melhores Multiplicadores"][0],
        "mul_2": res["Melhores Multiplicadores"][1],
        "Taxa de Evaporadação": res["Lista Vapor Entrada por Efeito (kg/h)"]
    })

df_results = pd.DataFrame(results)

# Mostrar tabela ao usuário
try:
    # caas_jupyter_tools.display_dataframe_to_user se disponível
    from caas_jupyter_tools import display_dataframe_to_user
    display_dataframe_to_user("Variação Pressão vs Consumo e Brix", df_results)
except Exception:
    display = df_results.head(20)
    print(display.to_string(index=False))

# Gráfico 1: Consumo total de vapor x Pressão
plt.figure(figsize=(8,4))
plt.plot(df_results["press_vapor"], df_results["vapor entrada"], marker='o')
plt.title("Consumo total de vapor (kg/h) vs Pressão do vapor vivo (bar)")
plt.xlabel("Pressão do vapor vivo (bar)")
plt.ylabel("Consumo total de vapor (kg/h)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico 2: Brix final x Pressão
plt.figure(figsize=(8,4))
plt.plot(df_results["press_vapor"], df_results["brix_final"], marker='o')
plt.title("Brix final vs Pressão do vapor vivo (bar)")
plt.xlabel("Pressão do vapor vivo (bar)")
plt.ylabel("Brix final (°Bx)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Salvar resultados em variável para o usuário
df_results