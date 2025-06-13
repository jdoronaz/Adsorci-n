import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Título
st.title("Simulación de Adsorción en un Tanque Agitado Continuo")

# Parámetros operativos
st.sidebar.header("Parámetros del sistema")
V = st.sidebar.number_input("Volumen del reactor (L)", value=0.5, step=0.1)
e = st.sidebar.number_input("Porosidad (-)", value=0.75, step=0.05)
H = st.sidebar.number_input("Caudal de la solución de alimentación (L/h)", value=1.0, step=0.1)
yF = st.sidebar.number_input("Concentración de entrada (g/L)", value=0.64, step=0.02)
a = st.sidebar.number_input("Superficie del adsorbente por unidad de volumen del tanque  (1/dm)", value=1.3, step=0.1)
k = st.sidebar.number_input("Coeficiente de tranasferencia de materia (dm/h)", value=25.0, step=5.0)
N = st.sidebar.number_input("Número de puntos (-)", value=1000, step=100)

# Selección de isoterma
st.sidebar.header("Isoterma de adsorción")
isoterma = st.sidebar.selectbox("Tipo de isoterma", ["Lineal", "Langmuir", "Freundlich"])

# Parámetros de la isoterma
if isoterma == "Lineal":
    K = st.sidebar.number_input("Constante K (-)", value=30.0, step=5.0)
    usar_teorica = st.sidebar.checkbox("Agregar solución analítica")

elif isoterma == "Langmuir":
    qmax = st.sidebar.number_input("qmax (g/L)", value=70.0, step=5.0)
    KL = st.sidebar.number_input("Constante KL (g/L)", value=50.0, step=5.0)


elif isoterma == "Freundlich":
    KF = st.sidebar.number_input("KF (g L)^(1-n)", value=50.0, step=5.0)
    n = st.sidebar.number_input("n (-)", value=0.65, step=0.05)

# Tiempo de simulación
t = np.linspace(0, 10, N)  # t_final = 10 h

# Modelo de tanque agitado continuo con equilibrio instantáneo
def simular_adsorcion(V, e, H, yF, a, k):
    y = np.zeros_like(t)
    y[0]=0
    q = np.zeros_like(t)

    dt = t[1] - t[0]
    alpha1 = H*dt/(e*V)
    alpha2 = k*a*dt/(e)
    alpha3 = k*a*dt/(1-e)

    for i in range(0, len(t)-1):
        if isoterma == "Lineal":
            y[i+1]=y[i]+alpha1*(yF-y[i])-alpha2*(y[i]-q[i]/K)
            q[i+1]=q[i]+alpha3*(y[i]-q[i]/K)

        elif isoterma == "Langmuir":
            y[i+1]=y[i]+alpha1*(yF-y[i])-alpha2*(y[i]-q[i]*KL/(qmax-q[i]))
            q[i+1]=q[i]+alpha3*(y[i]-q[i]*KL/(qmax-q[i]))

        elif isoterma == "Freundlich":
            y[i+1]=y[i]+alpha1*(yF-y[i])-alpha2*(y[i]-(q[i]/KF)**(1/n))
            q[i+1]=q[i]+alpha3*(y[i]-(q[i]/KF)**(1/n))

    return y, q

# Simulación
y, q = simular_adsorcion(V, e, H, yF, a, k)

# Inicializar las curvas teóricas si corresponde
y_teo = q_teo = None
if isoterma == "Lineal" and 'usar_teorica' in locals() and usar_teorica:
    b = H/(e*V) + k*a*(1+e/((1-e)*K))
    b2 = H/(e*V)
    sig1 = (1/2)*(b+np.sqrt((b**2)-4*k*a*H/(K*V*(1-e))))
    sig2 = (1/2)*(b-np.sqrt((b**2)-4*k*a*H/(K*V*(1-e))))
    
    y_teo = yF*(1-(b2-sig2)*np.exp(-sig1*t)/(sig1-sig2)-(sig1-b2)*np.exp(-sig2*t)/(sig1-sig2))
    q_teo = yF*K*(1+sig2*np.exp(-sig1*t)/(sig1-sig2)-sig1*np.exp(-sig2*t)/(sig1-sig2))
    
    # Calcular MAPE solo donde y_teo y q_teo sean distintos de cero (para evitar división por cero)
    mape_y = np.mean(np.abs((y - y_teo)[y_teo != 0] / y_teo[y_teo != 0])) * 100
    mape_q = np.mean(np.abs((q - q_teo)[q_teo != 0] / q_teo[q_teo != 0])) * 100

    # Mostrar en la app
    st.markdown("### Error entre solución simulada y analítica (Lineal)")
    st.write(f"**MAPE en y**: {mape_y:.2f} %")
    st.write(f"**MAPE en q**: {mape_q:.2f} %")

# Gráfico 2D con ejes primario y secundario
st.subheader("Perfiles de concentración")

fig, ax1 = plt.subplots()

# Eje primario: concentración en solución (y)
color = 'tab:blue'
ax1.set_xlabel("Tiempo (h)")
ax1.set_ylabel("Concentración en solución y (g/L)", color=color)
ax1.plot(t, y, color=color, marker='o', linestyle='', label='y (simulado)', markersize=3)
ax1.tick_params(axis='y', labelcolor=color)

if y_teo is not None:
    ax1.plot(t, y_teo, color='black', linestyle='--', label='yA (analítico)')  # línea sin símbolo
ax1.tick_params(axis='y', labelcolor=color)



# Eje secundario: concentración adsorbida (q)
ax2 = ax1.twinx()  # crear segundo eje y
color = 'tab:red'
ax2.set_ylabel("Concentración adsorbida q (g/L)", color=color)
ax2.plot(t, q, color=color, marker='s', linestyle='', label='q (simulado)', markersize=3)
ax2.tick_params(axis='y', labelcolor=color)

if q_teo is not None:
    ax2.plot(t, q_teo, color='black', linestyle='-', label='qA (analítico)')  # línea sin símbolo
ax2.tick_params(axis='y', labelcolor=color)

ax1.yaxis.grid(True, which='major', linestyle='-', linewidth=0.75, color='gray')  # líneas horizontales principales

# Título y leyenda
# Mostrar leyenda solo si se calcularon las curvas analíticas
if y_teo is not None and q_teo is not None:
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

fig.tight_layout()
st.pyplot(fig)



