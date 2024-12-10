from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_folium import st_folium
from sqlalchemy import create_engine, text
from keras.saving import load_model as load_model_keras
import folium
import streamlit as st
from functools import lru_cache
import plotly.express as px
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from concurrent.futures import ThreadPoolExecutor
from joblib import load
import joblib
import mysql.connector
import geopandas as gpd
import json
import os

import time



def main():

    st.set_page_config(page_title="Análisis Climatológico", layout="wide")

    database_password = st.secrets["database_password"]

    # Create the connection string
    engine = create_engine(f'mysql+pymysql://root:{database_password}@localhost:3306/AEMET')

    def run_query(query, params=None):
        with engine.connect() as connection:
            result = connection.execute(text(query), params)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df



    menu = ['Inicio','Valores climatológicos por comunidad y provincia', 'Comparador de valores climatológicos',
            'Mapa coroplético','Predicción del tiempo','Facebook Prophet','Diagrama MYSQL: Base de datos','About us']

    choice = st.sidebar.selectbox("Selecciona una opción", menu, key="menu_selectbox_unique")

    if choice == "Inicio":
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://facuso.es/wp-content/uploads/2023/09/6de3b76f2eeed4e2edfa5420ad9630bd.jpg" 
                     alt="Imagen oficial de la AEMET" 
                     width="250">
                <p>Imagen oficial de la AEMET</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.title("🌍 Un Viaje a Través del Clima y la Historia de España")
        st.markdown(
            "### 🌤️ Bienvenido/a a la Plataforma de Exploración de Datos de la AEMET (Agencia Estatal Meteorológica de España)."
        )

        st.markdown("""
        **Bienvenidos a España**, un país donde la historia y la naturaleza se entrelazan en un vibrante mosaico cultural. Desde los antiguos íberos hasta la influencia romana y la rica herencia musulmana, cada rincón de este territorio cuenta una historia única que ha dado forma a su identidad.
        """)

        st.header("Climas que Cuentan Historias")

        st.subheader("🌧️ El Norte: Un Abrazo Oceánico")
        st.markdown("""
        En el verde y montañoso norte, donde Galicia se asienta, el clima oceánico ofrece inviernos suaves y veranos frescos. Las lluvias son generosas, alimentando bosques frondosos y una rica biodiversidad. Aquí, las tradiciones se celebran con festivales que honran la conexión con la tierra y el mar.
        """)

        st.subheader("☀️ El Centro: Contrastes Cálidos")
        st.markdown("""
        A medida que nos adentramos en el corazón de España, encontramos el clima continental. En Madrid, las temperaturas pueden oscilar drásticamente entre el calor del verano y el frío del invierno. Esta variabilidad ha forjado una cultura vibrante y dinámica, donde la vida urbana se mezcla con la historia monumental.
        """)

        st.subheader("🏜️ El Sur: Sol y Sabor Mediterráneo")
        st.markdown("""
        En el sur, Andalucía brilla con su clima mediterráneo. Los veranos son cálidos y secos, perfectos para cultivar olivos y cítricos. Las fiestas flamencas y las tapas son solo algunas de las delicias que esta región tiene para ofrecer, reflejando la alegría de su gente.
        """)

        st.subheader("🏝️ Islas Canarias: Un Paraíso Subtropical")
        st.markdown("""
        Por último, las Islas Canarias, donde el clima subtropical permite disfrutar de temperaturas agradables durante todo el año. Este archipiélago es un refugio para quienes buscan sol y naturaleza, con paisajes volcánicos y playas de ensueño.
        """)

        st.markdown("""
        ### **Explora la Diversidad de España**
        Cada región de España no solo tiene su propio clima, sino también su propia historia, cultura y gastronomía. ¡Sumérgete en esta rica diversidad y descubre todo lo que España tiene para ofrecer!
        """)

        st.markdown(
            "🗺️ A tu izquierda encontrarás varias secciones, cada una con una breve introducción que te ayudará a navegar por la información."
        )

        st.markdown(
            "📊 En este sitio, podrás explorar y comparar datos históricos desde 2014, brindándote una visión profunda del clima en nuestro país."
        )
        # Crear 3 columnas para las imágenes
        col1, col2, col3 = st.columns(3)

        # Añadir imágenes a cada columna con un tamaño mayor
        with col1:
            st.image(
                "https://i.pinimg.com/originals/73/93/14/739314e72faa8f68bc12a29dcf0ce07c.jpg",
                caption="Ordesa y Monte Perdido",
                width=450  # Ajusta el ancho según sea necesario
            )
            st.image(
                "https://fascinatingspain.com/wp-content/uploads/benasque_nieve.jpg",
                caption="Benasque",
                width=450  # Ajusta el ancho según sea necesario
            )

        with col2:
            st.image(
                "https://www.viajes.com/blog/wp-content/uploads/2021/09/sea-6580532_1920.jpg",
                caption="Galicia, tierra de Meigas",
                width=450  # Ajusta el ancho según sea necesario
            )
            st.image(
                "https://i.pinimg.com/originals/cd/14/c8/cd14c8b90c06f714899d0d17e7d7fcd4.jpg",
                caption="Mallorca, Cala Egos - Cala d'Or",
                width=400  # Ajusta el ancho según sea necesario
            )

        with col3:
            st.image(
                "https://palenciaturismo.es/system/files/Monta%C3%B1aPalentinaGaleria5.jpg",
                caption="Palencia",
                width=450  # Ajusta el ancho según sea necesario
            )
            st.image(
                "https://i.pinimg.com/originals/d8/3a/f2/d83af2c8d615f0a8393ef3eeb9325435.jpg",
                caption="Asturias",
                width=450  # Ajusta el ancho según sea necesario
            )

    if choice == "Valores climatológicos por comunidad y provincia":

        # Título de la aplicación
        st.title("🌟 **Análisis Climatológico Interactivo por Ciudad** 🌤️")

        # Descripción de la aplicación
        st.markdown("""
        ### 🏙️ **Explora el Clima en Profundidad** 🌍

        Bienvenido al **Análisis Climatológico Interactivo**, donde podrás descubrir cómo ha evolucionado el clima en diferentes **ciudades** de **España** a lo largo del tiempo. 🌦️

        Selecciona una **ciudad** y un **rango de fechas** para visualizar datos detallados sobre las **temperaturas promedio**, **máximas y mínimas**, **precipitación**, **viento**, **altitud**, **humedad** y más. 📊

        🔍 **¿Qué puedes explorar aquí?**  
        - **Temperaturas media, máxima y mínima** a lo largo del tiempo 🌡️  
        - **Precipitación acumulada** (lluvias y otras condiciones meteorológicas) 🌧️  
        - **Velocidad y dirección del viento** 🌬️  
        - **Altitud** de la ciudad y su relación con el clima ⛰️
        - **Análisis visual** mediante gráficos interactivos 📈

        ¡Comienza a explorar el clima de tu ciudad favorita! 🏙️🌦️
        """)

        # Consulta de las ciudades disponibles
        ciudades_df = run_query("SELECT * FROM ciudades")

        # Sección de selección de parámetros
        with st.container():
            col1, col2, col3 = st.columns([1, 1, 1])

            # Sección de selección de parámetros
            with col1:
                ciudad_seleccionada = st.selectbox("🌆 Selecciona una ciudad", ciudades_df['ciudad'].tolist())
                st.write(f"**Datos climáticos para:** {ciudad_seleccionada}")
                ciudad_id = ciudades_df.loc[ciudades_df['ciudad'] == ciudad_seleccionada, 'ciudad_id'].values[0]
                # Obtener provincia_id de la ciudad seleccionada
                ciudad_info = run_query(f"""
                    SELECT p.provincia_id, c.ciudad_id, c.ciudad
                    FROM ciudades c
                    JOIN provincias p ON c.ciudad_id = p.provincia_id
                    WHERE c.ciudad = '{ciudad_seleccionada}'
                """)

                # Verificar si la consulta devuelve resultados
                if not ciudad_info.empty:
                    provincia_id = ciudad_info['provincia_id'].values[0]
                    ciudad_id = ciudad_info['ciudad_id'].values[0]
                    ciudad = ciudad_info['ciudad'].values[0]
                    st.write(f"**Provincia ID**: {provincia_id} - **Ciudad ID**: {ciudad_id} - **Ciudad**: {ciudad}")
                else:
                    st.warning(f"No se encontraron datos para la ciudad '{ciudad_seleccionada}'.")

                # Obtener nombre de la provincia
                provincia_info = run_query(f"""
                    SELECT provincia
                    FROM provincias
                    WHERE provincia_id = {provincia_id}
                """)
                provincia = provincia_info['provincia'].values[0]

                # Mostrar la comunidad y la provincia
                st.write(f"🌍 📍 **Provincia**: {provincia}")

            with col2:
                fecha_inicio = st.date_input("📅 Fecha de inicio:", value=datetime(2014, 1, 1))

            with col3:
                fecha_fin = st.date_input("📅 Fecha de fin:", value=datetime(2024, 10, 31))

        # Consulta de datos climáticos
        query = f"""
            SELECT fecha, tmed, tmax, tmin, prec, velemedia, dir, hrMedia, altitud, hrMax, hrMin
            FROM valores_climatologicos
            WHERE nombre_id = {ciudad_id} AND fecha BETWEEN '{fecha_inicio}' AND '{fecha_fin}'
        """
        datos_climaticos_df = run_query(query)

        # Comprobación si hay datos
        if not datos_climaticos_df.empty:
            st.subheader("📊 Datos Climáticos")
            st.dataframe(datos_climaticos_df, use_container_width=True)

            # Gráficos interactivos
            with st.container():
                st.markdown("### 🌡️ Visualización General")
                col1, col2 = st.columns(2)

                # Gráfico de Temperatura Media
                with col1:
                    st.markdown("#### 📈 Temperatura Media")
                    fig_temp = go.Figure(go.Scatter(
                        x=datos_climaticos_df['fecha'],
                        y=datos_climaticos_df['tmed'],
                        mode='lines',
                        line=dict(color='red'),
                        name='Temperatura Media'
                    ))
                    fig_temp.update_layout(title="Temperatura Media", xaxis_title="Fecha", yaxis_title="°C")
                    st.plotly_chart(fig_temp, use_container_width=True)

                # Gráfico de Velocidad del Viento
                with col2:
                    st.markdown("#### 💨 Velocidad del Viento")
                    fig_wind = go.Figure(go.Scatter(
                        x=datos_climaticos_df['fecha'],
                        y=datos_climaticos_df['velemedia'],
                        mode='lines',
                        line=dict(color='green'),
                        name='Velocidad del Viento'
                    ))
                    fig_wind.update_layout(title="Velocidad Media del Viento", xaxis_title="Fecha", yaxis_title="km/h")
                    st.plotly_chart(fig_wind, use_container_width=True)

            with st.container():
                col3, col4 = st.columns(2)

                # Gráfico de Precipitación
                with col3:
                    st.markdown("#### 🌧️ Precipitación")
                    fig_precip = go.Figure(go.Bar(
                        x=datos_climaticos_df['fecha'],
                        y=datos_climaticos_df['prec'],
                        marker_color='darkblue',
                        name='Precipitación'
                    ))
                    fig_precip.update_layout(title="Precipitación Acumulada", xaxis_title="Fecha", yaxis_title="mm")
                    st.plotly_chart(fig_precip, use_container_width=True)

                # Gráfico de Humedad Relativa
                with col4:
                    st.markdown("#### 💧 Humedad Relativa")
                    fig_humidity = go.Figure(go.Scatter(
                        x=datos_climaticos_df['fecha'],
                        y=datos_climaticos_df['hrMedia'],
                        mode='lines',
                        line=dict(color='lightblue'),
                        name='Humedad Relativa'
                    ))
                    fig_humidity.update_layout(title="Humedad Relativa Media", xaxis_title="Fecha", yaxis_title="%")
                    st.plotly_chart(fig_humidity, use_container_width=True)

            with st.container():
                col5, col6 = st.columns(2)

                # Gráfico de Altitud
                with col5:
                    st.markdown("#### ⛰️ Altitud")
                    fig_altitud = go.Figure(go.Scatter(
                        x=datos_climaticos_df['fecha'],
                        y=datos_climaticos_df['altitud'],
                        mode='lines',
                        line=dict(color='orange'),
                        name='Altitud'
                    ))
                    fig_altitud.update_layout(title="Altitud de la Ciudad", xaxis_title="Fecha",
                                              yaxis_title="m sobre el nivel del mar")
                    st.plotly_chart(fig_altitud, use_container_width=True)

                # Gráfico de Temperatura Máxima y Mínima
                with col6:
                    st.markdown("#### 🌡️ Temperaturas Máximas y Mínimas")
                    fig_temp_max_min = go.Figure()

                    # Añadir temperatura máxima
                    fig_temp_max_min.add_trace(go.Scatter(
                        x=datos_climaticos_df['fecha'],
                        y=datos_climaticos_df['tmax'],
                        mode='lines',
                        name='Temperatura Máxima',
                        line=dict(color='red', dash='solid')
                    ))

                    # Añadir temperatura mínima
                    fig_temp_max_min.add_trace(go.Scatter(
                        x=datos_climaticos_df['fecha'],
                        y=datos_climaticos_df['tmin'],
                        mode='lines',
                        name='Temperatura Mínima',
                        line=dict(color='blue', dash='solid')
                    ))

                    fig_temp_max_min.update_layout(title="Temperaturas Máximas y Mínimas", xaxis_title="Fecha",
                                                   yaxis_title="°C")
                    st.plotly_chart(fig_temp_max_min, use_container_width=True)

            # Consultas avanzadas (más métricas)
            st.markdown("### 📊 Consultas Avanzadas")
            queries = {
                "Temperaturas Máxima y Mínima Diaria": f"""
                    SELECT fecha, MAX(tmax) AS max_temperature, MIN(tmin) AS min_temperature 
                    FROM valores_climatologicos 
                    WHERE nombre_id = {ciudad_id} AND fecha BETWEEN '{fecha_inicio}' AND '{fecha_fin}'
                    GROUP BY fecha ORDER BY fecha;
                """,
                "Dirección del Viento Promedio": f"""
                    SELECT fecha, AVG(dir) AS average_wind_direction
                    FROM valores_climatologicos
                    WHERE nombre_id = {ciudad_id} AND fecha BETWEEN '{fecha_inicio}' AND '{fecha_fin}'
                    GROUP BY fecha ORDER BY fecha;
                """,
                "Precipitación Total Mensual": f"""
                    SELECT DATE_FORMAT(fecha, '%Y-%m') AS month, SUM(prec) AS total_precipitation 
                    FROM valores_climatologicos 
                    WHERE nombre_id = {ciudad_id} AND fecha BETWEEN '{fecha_inicio}' AND '{fecha_fin}'
                    GROUP BY month ORDER BY month;
                """,
            }

            selected_query = st.selectbox("🔍 Selecciona una consulta avanzada:", list(queries.keys()))
            data_avanzada = run_query(queries[selected_query])

            if not data_avanzada.empty:
                fig_advanced = go.Figure()
                for col in data_avanzada.columns[1:]:
                    fig_advanced.add_trace(go.Scatter(
                        x=data_avanzada['fecha'],
                        y=data_avanzada[col],
                        mode='lines',
                        name=col
                    ))
                fig_advanced.update_layout(title=f"Análisis Avanzado - {selected_query}", xaxis_title="Fecha",
                                           yaxis_title="Valor")
                st.plotly_chart(fig_advanced, use_container_width=True)
        else:
            st.warning("⚠️ No se encontraron datos para el rango de fechas seleccionado.")






    if choice == "Comparador de valores climatológicos":
        # Título de la aplicación con un toque visual
        st.title("🌤️ **Comparativa de los Valores Climatológicos** 🌡️")

        # Breve explicación de la aplicación con emojis y formato atractivo
        st.markdown("""
        ✨ **¡Bienvenido a la Comparativa Climática!** ✨  
        Esta aplicación te permite **comparar los valores climatológicos** de diferentes provincias entre dos años específicos. 🌍

        ### ¿Qué puedes hacer aquí? 🤔
        - **Selecciona una o dos provincias** de las disponibles 🏞️
        - **Elige dos años** y descubre cómo ha variado el clima entre ellos 📅
        - Compara las **temperaturas medias, máximas y mínimas**, además de otros factores como:
            - **Precipitación** 🌧️
            - **Dirección del Viento** 🌬️
            - **Velocidad del Viento** 💨
            - **Humedad Relativa** 💧

        **Explora los datos y descubre patrones climáticos interesantes!** 🔍  
        Cada gráfico te mostrará la evolución de estos factores a lo largo de los dos años seleccionados, de manera clara y visual. 📊

        👉 ¡Selecciona una o dos provincias y dos años para comenzar!
        """)

        # Cacheo de la función para reducir las consultas a la base de datos
        @lru_cache(maxsize=32)
        def load_data_from_db(provincia_ids, year1, year2):
            # Si solo se selecciona una provincia, agregamos un filtro por la provincia
            query = f"""
                SELECT 
                    vc.Fecha, 
                    vc.Tmed, 
                    vc.prec, 
                    vc.tmin, 
                    vc.tmax, 
                    vc.dir, 
                    vc.velemedia, 
                    vc.hrMedia, 
                    p.provincia,
                    p.provincia_id
                FROM valores_climatologicos vc
                JOIN provincias p ON vc.Provincia_id = p.provincia_id
                WHERE vc.Provincia_id IN ({', '.join(map(str, provincia_ids))})
                AND (YEAR(vc.Fecha) = {year1} OR YEAR(vc.Fecha) = {year2})
            """
            return pd.read_sql(query, engine)

        # Subtítulo de la comparación
        st.subheader("🌍 Comparación de la Temperatura y Otros Factores por Provincias 🏞️")

        # Selección de una o dos provincias
        provincias_df = pd.read_sql("SELECT * FROM provincias", engine)
        provincia1 = st.selectbox("🔎 Selecciona la primera provincia", provincias_df["provincia"].tolist())

        # Crear un diccionario de IDs de provincias para acceso rápido
        provincia_dict = dict(zip(provincias_df['provincia'], provincias_df['provincia_id']))
        # Selección de la primera provincia con un key único
        provincia1 = st.selectbox("🔎 Selecciona la primera provincia", provincias_df["provincia"].tolist(),
                                  key="provincia1")

        # Preguntar si se quiere comparar con una segunda provincia
        comparar_dos = st.checkbox("¿Comparar con otra provincia?", key="comparar_dos")

        # Determinar los IDs de las provincias seleccionadas
        if comparar_dos:
            # Si se compara con otra provincia, se muestra la lista sin la provincia ya seleccionada
            provincia2 = st.selectbox("🔎 Selecciona la segunda provincia",
                                      [provincia for provincia in provincias_df['provincia'] if
                                       provincia != provincia1],
                                      key="provincia2")

            # Crear lista de IDs de provincias
            provincia_ids = [provincia_dict[provincia1], provincia_dict[provincia2]]
        else:
            provincia_ids = [provincia_dict[provincia1]]

        # Selección de los años con keys únicos
        year1 = st.selectbox("📅 Selecciona el primer año",
                             [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024], key="year1")
        year2 = st.selectbox("📅 Selecciona el segundo año",
                             [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024], key="year2")

        # Preguntar qué factores el usuario quiere comparar
        factores = st.multiselect("📈 ¿Qué factores deseas comparar?",
                                  ['Temperatura Media', 'Temperatura Mínima', 'Temperatura Máxima',
                                   'Precipitación', 'Dirección del Viento', 'Velocidad del Viento', 'Humedad Relativa'])

        # Botón para cargar los datos y mostrar gráficos
        cargar_datos = st.button("🔄 Cargar Datos y Mostrar Gráficos")

        if cargar_datos:
            # Cargar los datos desde la base de datos con optimización
            data = load_data_from_db(tuple(provincia_ids), year1, year2)

            # Calcular estadísticas de las temperaturas y otros factores
            data['Year'] = pd.to_datetime(data['Fecha']).dt.year
            stats = data.groupby(['Year', 'Fecha', 'provincia']).agg(
                Tmed_min=('Tmed', 'min'),
                Tmed_max=('Tmed', 'max'),
                Tmed_median=('Tmed', 'median'),
                Prec_mean=('prec', 'mean'),
                Tmin_mean=('tmin', 'mean'),
                Tmax_mean=('tmax', 'mean'),
                Dir_mean=('dir', 'mean'),
                Velemedia_min=('velemedia', 'min'),
                Velemedia_max=('velemedia', 'max'),
                HrMedia_min=('hrMedia', 'min'),
                HrMedia_max=('hrMedia', 'max')
            ).reset_index()

            # Filtrar los factores seleccionados para mostrar
            factores_dict = {
                'Temperatura Media': ['Tmed_min', 'Tmed_max', 'Tmed_median'],
                'Temperatura Mínima': ['Tmed_min'],
                'Temperatura Máxima': ['Tmed_max'],
                'Precipitación': ['Prec_mean'],
                'Dirección del Viento': ['Dir_mean'],
                'Velocidad del Viento': ['Velemedia_min', 'Velemedia_max'],
                'Humedad Relativa': ['HrMedia_min', 'HrMedia_max']
            }

            selected_factors = []
            for factor in factores:
                selected_factors.extend(factores_dict[factor])

            st.write("### 📉 Estadísticas de Factores Climáticos Seleccionados:")
            st.write(stats[['Fecha', 'provincia'] + selected_factors])

            # Gráfico de Comparación de Factores Climáticos
            st.write("### 📊 Gráfico de Comparación Climática")

            fig, ax = plt.subplots(figsize=(16, 8))

            # Graficar los factores seleccionados
            for provincia in stats['provincia'].unique():
                provincia_data = stats[stats['provincia'] == provincia]
                for factor in selected_factors:
                    ax.plot(provincia_data['Fecha'], provincia_data[factor], label=f'{factor} - {provincia}',
                            marker='o')

            # Personalización del gráfico
            ax.set_title(
                f'📈 Comparación Climática entre {", ".join([provincia1, provincia2] if comparar_dos else [provincia1])} en {year1} y {year2}',
                fontsize=16)
            ax.set_xlabel('Fecha 📅', fontsize=14)
            ax.set_ylabel('Valor Climático', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True)
            plt.xticks(rotation=45)

            # Mostrar el gráfico en la app
            st.pyplot(fig)








    if choice == "Mapa coroplético":

        st.title("📊 Mapa Coroplético: Histórico de Temperaturas Medias en España")
        st.subheader("Explora las temperaturas medias de España con filtros dinámicos.")
        st.info("""
            1. Filtra por años, meses y provincias para ver las temperaturas medias en cada provincia.
            2. Un **mapa coroplético** es una representación geográfica en la que las áreas del mapa se colorean según valores de una variable. En este caso, se visualizan las **temperaturas medias** de cada provincia, lo que te permite identificar patrones geográficos de temperatura a lo largo del tiempo.
            3. Puedes interactuar con el mapa, seleccionar diferentes fechas y provincias para obtener información precisa y detallada sobre el clima en cada región.
        """)

        def crear_mapa_choropleth(geojson, df, color_column, fill_color, legend_name):
            mapa = folium.Map(location=[40.4168, -3.7038], zoom_start=6)
            folium.Choropleth(
                geo_data=geojson,
                data=df,
                columns=["provincia", color_column],
                key_on="feature.properties.name",
                fill_color=fill_color,  # Colores según la temperatura
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=legend_name,
            ).add_to(mapa)
            return mapa

        def mostrar_grafico_temperatura(df, provincia, year, month):
            fig = px.line(df, x="mes", y="media_tmed_mensual",
                          title=f"Tendencia de Temperatura Media en {provincia} ({month}/{year})",
                          labels={"mes": "Mes", "media_tmed_mensual": "Temperatura Media (°C)"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

        year = st.selectbox("Selecciona el año:", [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
        month = st.selectbox("Selecciona el mes:", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        provincia_seek = st.selectbox("Selecciona la provincia:", [
            'STA. CRUZ DE TENERIFE', 'BARCELONA', 'SEVILLA', 'CUENCA', 'ZARAGOZA', 'ILLES BALEARS', 'VALENCIA',
            'ZAMORA', 'PALENCIA', 'CASTELLON', 'LAS PALMAS', 'MADRID', 'CANTABRIA', 'GRANADA', 'TERUEL', 'BADAJOZ',
            'A CORUÑA', 'ASTURIAS', 'TARRAGONA', 'ALMERIA', 'ALICANTE', 'CADIZ', 'TOLEDO', 'BURGOS', 'GIRONA', 'MALAGA',
            'JAEN', 'MURCIA', 'LLEIDA', 'HUESCA', 'ALBACETE', 'NAVARRA', 'CORDOBA', 'OURENSE', 'CIUDAD REAL',
            'GIPUZKOA',
            'MELILLA', 'LEON', 'CACERES', 'SALAMANCA', 'HUELVA', 'LA RIOJA', 'BIZKAIA', 'GUADALAJARA', 'VALLADOLID',
            'ARABA/ALAVA', 'PONTEVEDRA', 'SEGOVIA', 'SORIA', 'AVILA', 'CEUTA', 'LUGO', 'BALEARES'
        ])

        # Query para obtener los datos filtrados
        query = f"""
            SELECT 
                DATE_FORMAT(t1.fecha, '%Y-%m') AS mes, 
                ROUND(AVG(t1.tmed), 2) AS media_tmed_mensual,
                t1.provincia_id, 
                t2.provincia 
            FROM 
                valores_climatologicos t1 
            RIGHT JOIN 
                provincias t2 ON t1.provincia_id = t2.provincia_id
            WHERE
                YEAR(t1.fecha) = {year}
                AND MONTH(t1.fecha) = {month}
                AND t2.provincia = '{provincia_seek}'
            GROUP BY 
                mes, t1.provincia_id, t2.provincia;
        """
        df = run_query(query)

        # Cargar el archivo GeoJSON de las provincias españolas
        with open("spain-provinces.geojson", "r", encoding="utf-8") as file:
            geojson_spain = json.load(file)

        # Mapeo de nombres de provincias
        map_provincia = {
            "STA. CRUZ DE TENERIFE": "Santa Cruz De Tenerife", "BARCELONA": "Barcelona",
            "SEVILLA": "Sevilla", "CUENCA": "Cuenca", "ZARAGOZA": "Zaragoza", "ILLES BALEARS": "Illes Balears",
            'VALENCIA': "València/Valencia", 'ZAMORA': "Zamora", 'PALENCIA': "Palencia",
            'CASTELLON': "Castelló/Castellón", 'LAS PALMAS': "Las Palmas", 'MADRID': "Madrid",
            'CANTABRIA': "Cantabria", 'GRANADA': "Granada", 'TERUEL': "Teruel", 'BADAJOZ': "Badajoz",
            'A CORUÑA': "A Coruña", 'ASTURIAS': "Asturias", 'TARRAGONA': "Tarragona", 'ALMERIA': "Almería",
            'ALICANTE': "Alacant/Alicante", 'CADIZ': "Cádiz", 'TOLEDO': "Toledo", 'BURGOS': "Burgos",
            'GIRONA': "Girona", 'MALAGA': "Málaga", 'JAEN': "Jaén", 'MURCIA': "Murcia", 'LLEIDA': "Lleida",
            'HUESCA': "Huesca", 'ALBACETE': "Albacete", 'NAVARRA': "Navarra", 'CORDOBA': "Córdoba",
            'OURENSE': "Ourense", 'CIUDAD REAL': "Ciudad Real", 'GIPUZKOA': "Gipuzkoa/Guipúzcoa", 'MELILLA': "Melilla",
            'LEON': "León", 'CACERES': "Cáceres", 'SALAMANCA': "Salamanca", 'HUELVA': "Huelva",
            'LA RIOJA': "La Rioja", 'BIZKAIA': "Bizkaia/Vizcaya", 'GUADALAJARA': "Guadalajara",
            'VALLADOLID': "Valladolid", 'ARABA/ALAVA': "Araba/Álava", 'PONTEVEDRA': "Pontevedra",
            'SEGOVIA': "Segovia", 'SORIA': "Soria", 'AVILA': "Ávila", 'CEUTA': "Ceuta", 'LUGO': "Lugo",
            'BALEARES': "Illes Balears"
        }
        df["provincia"] = df["provincia"].map(map_provincia)

        # Visualizar la tabla con datos de temperaturas
        st.markdown("### Datos de temperaturas medias mensuales:")
        st.dataframe(df)

        # Crear y mostrar el mapa choropleth
        mapa_espana = crear_mapa_choropleth(geojson_spain, df, "media_tmed_mensual", "YlGnBu",
                                            "Temperatura Media Mensual (°C)")
        st_folium(mapa_espana, width=725)

        # Gráfico de tendencia de temperatura
        st.markdown("### Gráfico de Temperaturas Medias Mensuales:")
        mostrar_grafico_temperatura(df, provincia_seek, year, month)

        # Información adicional sobre la fecha seleccionada
        st.info("2. Mapa de España con Temperaturas Medias Diarias.")

        # Selección de fecha
        date = st.date_input("Selecciona una fecha", value=pd.to_datetime(f"2023-01-01"))
        dia = date.strftime('%Y-%m-%d')

        # Consulta para el clima en la fecha seleccionada
        query1 = f"""
            SELECT 
                t1.fecha, 
                ROUND(AVG(t1.tmed), 2) AS media_tmed, 
                t1.provincia_id, 
                t2.provincia 
            FROM 
                valores_climatologicos t1 
            RIGHT JOIN 
                provincias t2 ON t1.provincia_id = t2.provincia_id
            WHERE
                t1.fecha = "{date}"
            GROUP BY 
                t1.fecha, t1.provincia_id, t2.provincia;
        """
        df_daily = run_query(query1)
        df_daily = df_daily[["fecha", "media_tmed", "provincia"]]
        df_daily["provincia"] = df_daily["provincia"].map(map_provincia)

        # Mostrar el gráfico interactivo para la fecha seleccionada
        st.write(f"### Temperatura Media para el día {dia}:")
        st.dataframe(df_daily)

        # Mapa interactivo de temperaturas diarias
        mapa_espana_daily = crear_mapa_choropleth(geojson_spain, df_daily, "media_tmed", "YlOrRd",
                                                  "Temperatura Media Diaria (°C)")
        st_folium(mapa_espana_daily, width=725)





    if choice=="Predicción del tiempo":

        # Cargar el modelo con Streamlit cache para optimizar el tiempo de carga
        @st.cache_resource
        def load_model(file_path):
            return load_model_keras(file_path)

        # Función para cargar el modelo y el escalador según las rutas
        def cargar_modelo_y_escalador(model_path, scaler_path):
            try:
                # Verifica si el archivo de modelo y escalador existe
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    # Cargar el modelo Keras desde el archivo .keras
                    model = load_model(model_path)

                    # Cargar el escalador con joblib
                    scaler = joblib.load(scaler_path)

                    # Mensaje de éxito
                    st.success(f"✅ Modelo y escalador cargados exitosamente.")
                    return model, scaler
                else:
                    st.error("❌ El archivo de modelo o escalador no existe en las rutas especificadas.")
                    return None, None
            except Exception as e:
                st.error(f"⚠️ Error al cargar el modelo o escalador: {e}")
                return None, None

        # Función para cargar los indicativos disponibles en las carpetas
        def obtener_indicativos_disponibles(tipo_modelo):
            try:
                # Directorios donde se encuentran los modelos y escaladores
                MODELOS_PATH = f"modelos_{tipo_modelo}"
                ESCALADORES_PATH = f"scaler_{tipo_modelo}"

                # Obtener los nombres de los modelos y escaladores disponibles
                modelos_disponibles = [f.split("_")[-1].replace(".keras", "") for f in os.listdir(MODELOS_PATH) if
                                       f.endswith(".keras")]
                scalers_disponibles = [f.split("_")[-1].replace(".pkl", "") for f in os.listdir(ESCALADORES_PATH) if
                                       f.endswith(".pkl")]

                # Intersección entre modelos y escaladores disponibles
                indicativos_disponibles = sorted(set(modelos_disponibles) & set(scalers_disponibles))

                return indicativos_disponibles
            except Exception as e:
                st.error(f"⚠️ Error al obtener los indicativos disponibles: {e}")
                return []

        # Explicación breve para cada tipo de modelo
        def explicar_modelo(tipo_modelo):
            if tipo_modelo == "gru":
                st.write("### 🤖 GRU (Gated Recurrent Unit)")
                st.write("""
                El **GRU** es un tipo de red neuronal recurrente que es eficiente y eficaz para trabajar con secuencias de datos. 
                Tiene una estructura más simple que el **LSTM**, pero logra resultados muy similares. 
                Es especialmente útil para predecir series temporales y datos secuenciales.
                """)
            elif tipo_modelo == "lstm":
                st.write("### 📚 LSTM (Long Short-Term Memory)")
                st.write("""
                **LSTM** es un tipo de red neuronal recurrente que tiene la capacidad de aprender y recordar a largo plazo. 
                Es ideal para datos secuenciales y temporales, como series de tiempo, ya que puede capturar dependencias de largo plazo.
                """)
            elif tipo_modelo == "rnn":
                st.write("### 🔄 RNN (Red Neuronal Recurrente)")
                st.write("""
                **RNN** es un tipo básico de red neuronal que utiliza la información de la secuencia pasada para hacer predicciones. 
                Aunque es útil para problemas de predicción secuencial, a veces tiene dificultades con la memoria a largo plazo, lo cual ha sido mejorado en LSTM y GRU.
                """)

        # Interfaz de Streamlit
        def interfaz_streamlit():
            # Selector para el tipo de modelo
            tipo_modelo = st.selectbox("Selecciona el tipo de modelo:", ["GRU", "LSTM", "RNN"]).lower()

            # Explicar el modelo seleccionado con un emoticono y breve descripción
            explicar_modelo(tipo_modelo)

            # Obtener los indicativos disponibles para el tipo seleccionado
            indicativos_disponibles = obtener_indicativos_disponibles(tipo_modelo)

            if not indicativos_disponibles:
                st.error("❌ No se encontraron modelos y escaladores compatibles en las carpetas.")
                return

            # Seleccionar el indicativo
            indicativo = st.selectbox("Selecciona un indicativo:", indicativos_disponibles)

            # Definir las rutas del modelo y escalador para el indicativo seleccionado
            model_path = os.path.join(f"modelos_{tipo_modelo}", f"modelo_{indicativo}.keras")
            scaler_path = os.path.join(f"scaler_{tipo_modelo}", f"scaler_{tipo_modelo}_{indicativo}.pkl")

            # Cargar el modelo y el escalador
            if st.button(f"🔄 Cargar Modelo para Indicativo {indicativo} ({tipo_modelo.upper()})"):
                model, scaler = cargar_modelo_y_escalador(model_path, scaler_path)

                if model is not None and scaler is not None:
                    # Entrada de datos manual
                    st.write("📝 Proporciona los valores de las siguientes características:")
                    tmed = st.number_input("Temperatura media (tmed):", value=15.0, step=0.1)
                    prec = st.number_input("Precipitación (prec):", value=0.0, step=0.1)
                    tmin = st.number_input("Temperatura mínima (tmin):", value=10.0, step=0.1)
                    tmax = st.number_input("Temperatura máxima (tmax):", value=20.0, step=0.1)
                    dire = st.number_input("Dirección del viento (dir):", value=180.0, step=1.0)
                    velemedia = st.number_input("Velocidad media del viento (velemedia):", value=2.0, step=0.1)
                    hrMedia = st.number_input("Humedad relativa media (hrMedia):", value=60.0, step=1.0)
                    hrMax = st.number_input("Humedad relativa máxima (hrMax):", value=90.0, step=1.0)

                    def predict():
                        try:
                            # Crear el array de entrada
                            input_data = np.array([[tmed, prec, tmin, tmax, dire, velemedia, hrMedia, hrMax]])

                            # Escalar los datos
                            scaled_data = scaler.transform(input_data)

                            # Preparar los datos para la predicción (formato 3D)
                            if tipo_modelo == "gru":
                                # Para GRU: Reshape para que sea (samples, 1, features)
                                X_predict = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))  # GRU
                            else:
                                # Para LSTM y RNN: Reshape para que sea (samples, timesteps, features)
                                X_predict = scaled_data.reshape(
                                    (scaled_data.shape[0], scaled_data.shape[1], 1))  # LSTM y RNN

                            # Realizar predicción
                            prediction = model.predict(X_predict)
                            prediction_rescaled = scaler.inverse_transform(
                                np.concatenate([prediction, np.zeros((prediction.shape[0], scaled_data.shape[1] - 1))],
                                               axis=1)
                            )[:, 0]  # Solo la columna 'tmed' rescalada

                            # Mostrar el resultado
                            st.success(
                                f"🔮 La predicción de la temperatura media para el día siguiente es: {prediction_rescaled[0]:.2f}°C")
                        except Exception as e:
                            st.error(f"⚠️ Error al realizar la predicción: {e}")

                    # Botón para predecir
                    st.button("🔮 Predecir Temperatura Media", on_click=predict)

        # Ejecutar la interfaz de Streamlit
        if __name__ == "__main__":
            # Título de la aplicación
            st.title("🌡️ Predicción de Temperatura Media para Mañana")
            interfaz_streamlit()





    if choice=="Facebook Prophet":

        st.title("🌞 Predicciones Climatológicas 🌦️")
        st.subheader("🔮 Modelos Predictivos con Facebook Prophet 📊")
        st.write("""
            🚀 Bienvenido a nuestra herramienta de predicción climatológica.
             
            📈 Carga modelos preentrenados y predice la temperatura media de los próximos días, semanas o meses.
            🗓️ Elige el rango de tiempo y el modelo que más te interese para obtener las mejores predicciones.
        """)

        # Explicación sobre Facebook Prophet
        st.write("""
            ### 🤖 ¿Qué es Facebook Prophet?

            🔍 **Facebook Prophet** es una herramienta de predicción de series temporales desarrollada por Facebook. Se utiliza para pronosticar datos con tendencias diarias, semanales y anuales, y es capaz de manejar datos faltantes y estacionalidades. Es muy eficaz en la predicción de series temporales como la climatología, las ventas de productos o la demanda de energía.

            📊 **¿Cómo funciona?**
            Prophet utiliza un enfoque basado en modelos aditivos que incorpora componentes como:
            - Tendencias (cambios a largo plazo en los datos).
            - Estacionalidades (patrones repetitivos en los datos).
            - Festivos o eventos especiales (que pueden afectar a las predicciones).

            🔧 Este modelo es extremadamente flexible y fácil de usar, ideal para datos con variabilidad estacional y efectos de eventos. A diferencia de otros modelos tradicionales, Prophet es capaz de manejar irregularidades en los datos y puede ser ajustado por los usuarios de forma sencilla.

            📅 En esta herramienta, usamos **Facebook Prophet** para predecir las temperaturas medias en España y otras métricas climatológicas a partir de los datos históricos que cargamos.
        """)

        st.title("Predicción de Temperatura para Mañana")

        # Función para cargar el modelo
        @st.cache_resource
        def load_model(file_path):
            return joblib.load(file_path)

        # Carga de modelos preentrenados
        models = {
            "Modelo Semestral 📅": load_model("f_prophet_biannual.pkl"),
            "Modelo Trimestral 🏞️": load_model("f_prophet_quarterly.pkl"),
            "Modelo Mensual 🗓️": load_model("f_prophet_monthly.pkl"),
            "Modelo Semanal 📆": load_model("f_prophet_weekly.pkl"),
            "Modelo Diario 🌞": load_model("f_prophet_daily.pkl")
        }

        # Cargar datos
        query1 = "SELECT fecha, tmed FROM valores_climatologicos"
        data_real = run_query(query1)
        data_real.rename(columns={"fecha": "ds", "tmed": "y"}, inplace=True)

        # Selección del modelo
        model_choice = st.selectbox(
            "🔍 **Seleccione el modelo que desee utilizar:**",
            list(models.keys())
        )

        # Selección del rango de tiempo
        times = {"Mañana 🌅": 1, "Semana 📅": 7, "Quincena 🔜": 14, "Mes 🗓️": 30}
        times_choice = st.selectbox(
            "⏳ **Seleccione el rango de tiempo que desee predecir:**",
            list(times.keys())
        )

        # Función para generar fechas futuras usando pandas directamente
        def generar_fechas_futuras(ultima_fecha, periods, freq='D'):
            fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), periods=periods, freq=freq)
            return fechas_futuras

        # Función para mostrar predicciones
        def show_predictions(model, data_real, periods):
            # Generar fechas futuras
            ultima_fecha = data_real['ds'].max()
            future_dates = generar_fechas_futuras(ultima_fecha, periods)

            # Crear DataFrame para predicciones
            future = pd.DataFrame({'ds': future_dates})

            # Realizar predicción
            forecast = model.predict(future)

            # Redondear y formatear la columna de predicción
            forecast['yhat'] = forecast['yhat'].round(2).astype(str) + " ºC"

            # Mostrar resultados
            st.write("📊 **Predicciones de temperatura media:**")
            st.dataframe(forecast[['ds', 'yhat']].rename(columns={'ds': 'Fecha', 'yhat': 'Temperatura media'}))

            # Gráfica interactiva
            fig = go.Figure()

            # Agregar datos reales
            fig.add_trace(go.Scatter(x=data_real['ds'], y=data_real['y'], mode='markers', name='Datos reales 🌡️'))

            # Agregar predicciones
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'].str.replace(' ºC', '').astype(float),
                                     mode='lines', name='Predicción 🔮'))

            fig.update_layout(title="📈 Predicción de Temperatura 🌡️", xaxis_title="Fecha",
                              yaxis_title="Temperatura (°C)")

            st.plotly_chart(fig)

        # Botón para predecir
        if st.button("🚀 ¡Predecir!"):
            model = models[model_choice]
            show_predictions(model, data_real, times[times_choice])
            st.balloons()





    if choice=="Diagrama MYSQL: Base de datos":
        st.image(image="Esquema_AEMET.png",
                caption="Esquema de la base de datos AEMET",
                use_column_width=True)

        st.subheader("Esquema base de datos AEMET:")
        st.write("""El esquema de esta base de datos consta de 4 tablas de datos en la que la principal sería la tabla llamada valores climatológicos y de la que surgen otras tres tablas llamadas indicativo, ciudades y provincias."
                                "En la tabla principal podemos encontrar los siguientes datos:
    
                   Fecha: recoge la fecha de medición de los valores climatológicos.
    
                   Altitud: altitud de medición de estos valores.
    
                   Tmed: temperatura media recogida durante el día en grados centígrados.
    
                   Prec: precipitaciones acumuladas en milímetros, que equivale a un 1 litro de agua por metro cuadrado."
    
                   Tmin: temperatura mínima registrada en el día.
    
                   HoraTmin: registro de hora de temperatura mínima.
    
                   Tmax: temperatura máxima registrada en el día.
    
                   HoraTmax: registro de hora de temperatura máxima.
    
                   Dir: direccional predominante del viento, expresada en grados (0°-360°) o en puntos cardinales (N, NE, E, etc.). Esto señala de dónde viene el viento, no hacia dónde va.
    
                   Velemedia: se refiere a la velocidad media del viento, expresada generalmente en kilómetros por hora (km/h) o metros por segundo (m/s). Este valor representa la velocidad promedio del viento registrada en el día.
    
                   Racha: se refiere a la racha máxima de viento, que es la mayor velocidad instantánea del viento registrada en un periodo determinado.
    
                   Horaracha: registro de hora de Racha.
    
                   HrMedia: Humedad relativa media del día.
    
                   HrMax: Humedad máxima registrada en el día.
    
                   HoraHrMax: Hora de registro de la humedad máxima.
    
                   HrMin: Humedad mínima registrada en el día.
    
                   HoraHrMin: Hora de registro de la humedad mínima.
    
                   Indicativo_id: índice asignado al valor indicativo de estación meteorológica.
    
                   Ciudad_id: índice asignado al valor ciudad.
    
                   Provincia_id: índice asignado al valor provincia.""")



    if choice == "About us":
        st.title("📬 **Contacto y Desarrolladores**")
        st.subheader(
            "Este proyecto ha sido desarrollado por los alumnos del curso de Data Science & IA. A continuación, encontrarás los datos de contacto.")

        # Establecer el tamaño de las imágenes (más pequeñas para un diseño más elegante)
        size = (250, 250)  # Imagen más pequeña y profesional

        # Cargar las imágenes de los miembros del equipo
        estela_img = Image.open("Estela.jpg").resize(size)
        pablo_img = Image.open("Pablo Petidier.jpg").resize(size)

        # Crear dos columnas
        col1, col2 = st.columns(2)

        # Primera columna (Estela)
        with col1:
            st.image(estela_img, caption="Estela Mojena Ávila", use_column_width=False)
            st.markdown("**Estela Mojena Ávila**")
            st.markdown("**📧 Correo Electrónico:** [estelamojenaavila@gmail.com](mailto:estelamojenaavila@gmail.com)")
            st.markdown("**📞 Teléfono:** [+34 622 68 33 95](tel:+34622683395)")
            st.markdown("**💼 LinkedIn:** [Estela Mojena Ávila](https://www.linkedin.com/in/estela-mojena-avila/)")
            st.markdown("**💻 GitHub:** [Estela8](https://github.com/Estela8)")

        # Segunda columna (Pablo)
        with col2:
            st.image(pablo_img, caption="Pablo Petidier Smit", use_column_width=False)
            st.markdown("**Pablo Petidier Smit**")
            st.markdown("**📧 Correo Electrónico:** [petidiersmit@gmail.com](mailto:petidiersmit@gmail.com)")
            st.markdown("**📞 Teléfono:** [+34 624 10 85 03](tel:+34624108503)")
            st.markdown("**💼 LinkedIn:** [Pablo Petidier Smit](https://www.linkedin.com/in/pablopetidier/)")
            st.markdown("**💻 GitHub:** [ppswns1988](https://github.com/ppswns1988)")

        # Espacio adicional para separar la información
        st.markdown("---")

        # Descripción del proyecto de manera breve
        st.markdown("""
        **Descripción del Proyecto:**  
        Este proyecto ha sido desarrollado como parte del curso de Data Science & IA. Su objetivo es proporcionar un análisis interactivo y visual de datos climáticos históricos, permitiendo a los usuarios explorar el clima de diferentes ciudades y provincias a lo largo del tiempo.

        **Objetivos del Proyecto:**  
        - Visualización de datos climáticos históricos (temperaturas, precipitación, viento, humedad).
        - Provisión de herramientas de análisis interactivo para el usuario.
        - Desarrollo y despliegue de un proyecto basado en Python y Streamlit.

        **Tecnologías Utilizadas:**  
        - **Python**: para procesamiento de datos.
        - **Streamlit**: para la creación de interfaces web interactivas.
        - **Plotly**: para gráficos interactivos.
        - **MySQL**: como base de datos para almacenar la información climática.

        **Fecha de Creación:** Octubre 2024
        """)

        st.markdown("---")

        # Agradecimiento y Cierre
        st.markdown("""
        **Agradecimientos:**  
        Agradecemos el apoyo recibido durante el curso de Data Science & IA, así como a todos aquellos que contribuyeron al desarrollo y mejora de este proyecto.

        Si tienes alguna duda o deseas ponerte en contacto con nosotros, no dudes en escribirnos a través de los correos electrónicos proporcionados.

        """)






if __name__ == "__main__":
    main()