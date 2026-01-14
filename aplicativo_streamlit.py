#python -m streamlit run aplicativo_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(
    page_title="Detecção de Fraudes em Cartão de Crédito",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("xgb_fraude.pkl")
    return model

def main():
    st.title("Detecção de Fraudes em Transações de Cartão de Crédito")
    st.write(
        "Aplicativo de apoio à análise de transações suspeitas utilizando o modelo XGBoost "
        "treinado a partir do dataset público de fraudes em cartão de crédito."
    )

    df = load_data()
    model = load_model()

    tab_dashboard, tab_nova_transacao = st.tabs(["Dashboard de Dados", "Inserir Nova Transação"])

    # ----------------------- DASHBOARD -----------------------
    with tab_dashboard:
        st.subheader("Visão Geral do Dataset")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de transações", f"{len(df):,}".replace(",", "."))
        with col2:
            num_fraudes = int(df["Class"].sum())
            st.metric("Total de fraudes", f"{num_fraudes:,}".replace(",", "."))
        with col3:
            perc_fraude = 100 * df["Class"].mean()
            st.metric("Percentual de fraudes", f"{perc_fraude:.3f}%")

        st.markdown("---")
        
        st.subheader("Análise Detalhada das Distribuições")

        # ===== GRÁFICO 1: DISTRIBUIÇÃO DO VALOR (AMOUNT) =====
        st.markdown("#### Distribuição dos Valores das Transações")
        st.write(
            "A maioria das transações concentra-se em valores baixos, com alguns valores extremos. "
            "Fraudes não apresentam padrão significativamente diferente em relação aos valores."
        )
        
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.hist(
            df[df["Class"] == 0]["Amount"], 
            bins=50, 
            alpha=0.6, 
            label="Não Fraude", 
            color="#2E86AB",
            edgecolor="black"
        )
        ax1.hist(
            df[df["Class"] == 1]["Amount"], 
            bins=50, 
            alpha=0.6, 
            label="Fraude", 
            color="#A23B72",
            edgecolor="black"
        )
        ax1.set_xlabel("Valor da Transação (Amount)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Quantidade de Transações", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=10)
        ax1.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)

        # ===== GRÁFICO 2: DISTRIBUIÇÃO DO TEMPO (TIME) =====
        st.markdown("#### Distribuição do Tempo das Transações")
        st.write(
            "As transações não ocorrem uniformemente ao longo do período. "
            "Há janelas de horário com concentração maior de operações, sugerindo picos de uso."
        )
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.hist(
            df["Time"], 
            bins=50, 
            alpha=0.7, 
            color="#06A77D",
            edgecolor="black"
        )
        ax2.set_xlabel("Tempo desde primeira transação (segundos)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Quantidade de Transações", fontsize=11, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

        # ===== GRÁFICO 3: COMPARAÇÃO BOXPLOT COM ESCALA LOGARÍTMICA =====
        st.markdown("#### Comparação de Valores: Fraude vs Não Fraude")
        st.write(
            "A visualização utiliza escala logarítmica para melhor destacar as diferenças de distribuição, "
            "já que há grande concentração de valores baixos e outliers extremos. "
            "A mediana das fraudes tende a ser ligeiramente mais alta, sugerindo possível padrão."
        )
        
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        df_plot = df.copy()
        df_plot["Classe"] = df_plot["Class"].map({0: "Não Fraude", 1: "Fraude"})
        
        # Aplicar escala logarítmica para melhor visualização
        sns.boxplot(
            data=df_plot, 
            x="Classe", 
            y="Amount", 
            ax=ax3,
            palette={"Não Fraude": "#2E86AB", "Fraude": "#A23B72"},
            width=0.5
        )
        ax3.set_xlabel("Classificação da Transação", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Valor da Transação (Amount) - Escala Logarítmica", fontsize=11, fontweight="bold")
        ax3.set_yscale("log")
        ax3.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)

        st.markdown("---")
        st.subheader("Amostra de Dados Originais")

        n_linhas = st.slider("Número de linhas para visualizar", 5, 50, 10, step=5)
        st.dataframe(df.head(n_linhas), use_container_width=True)

    # ------------------- NOVA TRANSAÇÃO ----------------------
    with tab_nova_transacao:
        st.subheader("Classificar Transações via Upload de Arquivo")

        st.write(
            "Carregue um arquivo CSV contendo as transações que deseja classificar. "
            "O arquivo deve conter as mesmas colunas do dataset original (Time, V1-V28, Amount). "
            "O modelo XGBoost analisará cada linha e retornará a classificação (Fraude ou Não Fraude) "
            "com a probabilidade estimada."
        )

        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV",
            type="csv",
            help="Arquivo com colunas: Time, V1-V28, Amount"
        )

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                
                st.markdown("### Dados Carregados")
                st.dataframe(df_upload, use_container_width=True)
                
                # Validar se todas as colunas necessárias estão presentes
                feature_cols = [c for c in df.columns if c != "Class"]
                missing_cols = set(feature_cols) - set(df_upload.columns)
                
                if missing_cols:
                    st.error(
                        f"Arquivo incompleto. Colunas faltantes: {', '.join(missing_cols)}"
                    )
                else:
                    if st.button("Classificar Transações"):
                        try:
                            # Garantir ordem correta das colunas
                            df_upload_ordered = df_upload[feature_cols]
                            
                            # Realizar predição
                            predictions = model.predict(df_upload_ordered)
                            probabilities = model.predict_proba(df_upload_ordered)[:, 1]
                            
                            # Criar dataframe com resultados
                            df_results = df_upload.copy()
                            df_results["Predição"] = predictions.astype(int)
                            df_results["Classe_Predita"] = df_results["Predição"].map(
                                {0: "Não Fraude", 1: "Fraude"}
                            )
                            df_results["Probabilidade_Fraude"] = probabilities
                            df_results["Probabilidade_Fraude_Pct"] = (probabilities * 100).round(2)
                            
                            st.markdown("### Resultados da Classificação")
                            
                            # Resumo
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                total_processado = len(df_results)
                                st.metric("Total de transações processadas", total_processado)
                            with col2:
                                num_fraudes_pred = int((df_results["Predição"] == 1).sum())
                                st.metric("Transações classificadas como fraude", num_fraudes_pred)
                            with col3:
                                perc_fraudes_pred = 100 * num_fraudes_pred / len(df_results)
                                st.metric("Percentual de fraudes", f"{perc_fraudes_pred:.2f}%")
                            
                            st.markdown("---")
                            
                            # Tabela de resultados
                            st.markdown("#### Detalhamento de Resultados")
                            cols_display = ["Classe_Predita", "Probabilidade_Fraude_Pct"] + feature_cols
                            st.dataframe(
                                df_results[cols_display].rename(
                                    columns={"Probabilidade_Fraude_Pct": "Prob. Fraude (%)"}
                                ),
                                use_container_width=True
                            )
                            
                            # Opção para download
                            csv_results = df_results.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="Baixar Resultados (CSV)",
                                data=csv_results,
                                file_name="resultados_classificacao.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Erro ao classificar: {str(e)}")
            
            except Exception as e:
                st.error(f"Erro ao carregar arquivo: {str(e)}")
        else:
            st.info("Aguardando upload de arquivo CSV...")

if __name__ == "__main__":
    main()
