
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from apriori_algorithm import AprioriAlgorithm
import io

st.set_page_config(
    page_title="Analisis Pola Pembelian Mini Market",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Font Awesome */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #00bcd4 0%, #0097a7 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #00bcd4 0%, #0097a7 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #00bcd4 0%, #0097a7 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0, 188, 212, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #00bcd4 0%, #0097a7 100%);
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Success message */
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Download button custom */
    .download-btn {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        text-decoration: none;
        display: inline-block;
        font-weight: 600;
        margin: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1><i class="fas fa-shopping-cart"></i> Analisis Pola Pembelian Mini Market</h1>
        <p>Analisis Pola Pembelian Produk Makanan Ringan dengan Algoritma Apriori</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### <i class='fas fa-cog'></i> Pengaturan Parameter", unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload File CSV",
        type=['csv'],
        help="Upload file CSV dengan format: TransactionID, Items"
    )
    
    st.markdown("---")
    
    # Parameters
    st.markdown("### <i class='fas fa-chart-bar'></i> Parameter Apriori", unsafe_allow_html=True)
    
    min_support = st.slider(
        "Minimum Support (%)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Minimum support untuk frequent itemsets"
    )
    
    min_confidence = st.slider(
        "Minimum Confidence (%)",
        min_value=20,
        max_value=100,
        value=50,
        step=10,
        help="Minimum confidence untuk association rules"
    )
    
    st.markdown("---")
    
    # Info
    st.markdown("### <i class='fas fa-info-circle'></i> Informasi", unsafe_allow_html=True)
    st.info("""
    **Cara Penggunaan:**
    1. Upload file CSV
    2. Atur parameter support & confidence
    3. Klik 'Jalankan Analisis'
    4. Lihat hasil dan download report
    """)
    
    # Sample data download
    st.markdown("### <i class='fas fa-download'></i> Download Sample", unsafe_allow_html=True)
    with open('data_sample.csv', 'rb') as f:
        st.download_button(
            label="Download Contoh CSV",
            data=f,
            file_name="data_sample.csv",
            mime="text/csv",
            width="stretch"
        )

# Main content
if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Validate CSV format
        if 'TransactionID' not in df.columns or 'Items' not in df.columns:
            st.error("Format CSV tidak sesuai! Pastikan ada kolom 'TransactionID' dan 'Items'")
        else:
            # Display uploaded data preview
            with st.expander("Preview Data yang Diupload", expanded=True):
                st.dataframe(df, width="stretch")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label"><i class="fas fa-receipt"></i> Total Transaksi</div>
                            <div class="metric-value">{len(df)}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Get unique items
                    all_items = set()
                    for items in df['Items']:
                        item_list = [item.strip() for item in str(items).split(',')]
                        all_items.update(item_list)
                    
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label"><i class="fas fa-box"></i> Jenis Produk</div>
                            <div class="metric-value">{len(all_items)}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    avg_items = df['Items'].apply(lambda x: len(str(x).split(','))).mean()
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label"><i class="fas fa-shopping-basket"></i> Rata-rata Item/Transaksi</div>
                            <div class="metric-value">{avg_items:.1f}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Run analysis button
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Jalankan Analisis Apriori", width="stretch"):
                with st.spinner("Sedang menganalisis data..."):
                    # Prepare transactions
                    transactions = []
                    for items in df['Items']:
                        transaction = [item.strip() for item in str(items).split(',')]
                        transactions.append(transaction)
                    
                    # Run Apriori
                    apriori = AprioriAlgorithm(
                        min_support=min_support/100,
                        min_confidence=min_confidence/100
                    )
                    apriori.load_transactions(transactions)
                    apriori.find_frequent_itemsets()
                    apriori.generate_association_rules()
                    
                    # Store results in session state
                    st.session_state['apriori'] = apriori
                    st.session_state['analysis_done'] = True
                    
                    st.success("Analisis selesai!")
            
            # Display results if analysis is done
            if 'analysis_done' in st.session_state and st.session_state['analysis_done']:
                apriori = st.session_state['apriori']
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Tabs for results
                tab1, tab2, tab3 = st.tabs([
                    "Frequent Itemsets",
                    "Association Rules",
                    "Visualisasi"
                ])
                
                with tab1:
                    st.markdown("### <i class='fas fa-chart-bar'></i> Frequent Itemsets", unsafe_allow_html=True)
                    st.markdown("Kumpulan item yang sering muncul bersamaan dalam transaksi")
                    
                    freq_df = apriori.get_frequent_itemsets_df()
                    
                    if not freq_df.empty:
                        st.dataframe(
                            freq_df.style.background_gradient(cmap='Blues', subset=['Support']),
                            width="stretch"
                        )
                        
                        # Download button
                        csv = freq_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Frequent Itemsets (CSV)",
                            data=csv,
                            file_name="frequent_itemsets.csv",
                            mime="text/csv",
                            width="stretch"
                        )
                    else:
                        st.warning("Tidak ada frequent itemsets yang ditemukan. Coba turunkan nilai minimum support.")
                
                with tab2:
                    st.markdown("### <i class='fas fa-link'></i> Association Rules", unsafe_allow_html=True)
                    st.markdown("Aturan asosiasi: Jika membeli A, maka kemungkinan membeli B")
                    
                    rules_df = apriori.get_association_rules_df()
                    
                    if not rules_df.empty:
                        st.dataframe(
                            rules_df,
                            width="stretch"
                        )
                        
                        # Interpretation
                        st.markdown("---")
                        st.markdown("#### <i class='fas fa-lightbulb'></i> Interpretasi Metrik:", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("""
                                <div class="info-card">
                                    <h4><i class="fas fa-chart-bar"></i> Support</h4>
                                    <p>Seberapa sering kombinasi item muncul dalam semua transaksi</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("""
                                <div class="info-card">
                                    <h4><i class="fas fa-bullseye"></i> Confidence</h4>
                                    <p>Peluang membeli item B jika sudah membeli item A</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("""
                                <div class="info-card">
                                    <h4><i class="fas fa-rocket"></i> Lift</h4>
                                    <p>Seberapa kuat hubungan antar item (>1 = positif, <1 = negatif)</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Download button
                        csv = rules_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Association Rules (CSV)",
                            data=csv,
                            file_name="association_rules.csv",
                            mime="text/csv",
                            width="stretch"
                        )
                    else:
                        st.warning("⚠️ Tidak ada association rules yang ditemukan. Coba turunkan nilai minimum confidence.")
                
                with tab3:
                    st.markdown("### <i class='fas fa-chart-line'></i> Visualisasi Hasil", unsafe_allow_html=True)
                    
                    # Visualization 1: Top Frequent Items
                    st.markdown("#### <i class='fas fa-trophy'></i> Top 10 Item Paling Sering Dibeli", unsafe_allow_html=True)
                    
                    item_counts = {}
                    for transaction in transactions:
                        for item in transaction:
                            item_counts[item] = item_counts.get(item, 0) + 1
                    
                    top_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    if top_items:
                        fig = px.bar(
                            x=[item[1] for item in top_items],
                            y=[item[0] for item in top_items],
                            orientation='h',
                            labels={'x': 'Jumlah Transaksi', 'y': 'Produk'},
                            color=[item[1] for item in top_items],
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(
                            showlegend=False,
                            height=500,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Visualization 2: Support vs Confidence scatter
                    if not rules_df.empty:
                        st.markdown("#### <i class='fas fa-bullseye'></i> Support vs Confidence (Association Rules)", unsafe_allow_html=True)
                        
                        # Extract numeric values
                        support_values = [float(x.strip('%')) for x in rules_df['Support']]
                        confidence_values = [float(x.strip('%')) for x in rules_df['Confidence']]
                        
                        fig2 = go.Figure(data=go.Scatter(
                            x=support_values,
                            y=confidence_values,
                            mode='markers',
                            marker=dict(
                                size=10,
                                color=confidence_values,
                                colorscale='Plasma',
                                showscale=True,
                                colorbar=dict(title="Confidence %")
                            ),
                            text=[f"{rules_df['Antecedent (Jika)'].iloc[i]} → {rules_df['Consequent (Maka)'].iloc[i]}" 
                                  for i in range(len(rules_df))],
                            hovertemplate='<b>%{text}</b><br>Support: %{x:.2f}%<br>Confidence: %{y:.2f}%<extra></extra>'
                        ))
                        
                        fig2.update_layout(
                            xaxis_title="Support (%)",
                            yaxis_title="Confidence (%)",
                            height=500,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Visualization 3: Frequent Itemsets by Size
                    st.markdown("#### <i class='fas fa-box-open'></i> Distribusi Frequent Itemsets berdasarkan Ukuran", unsafe_allow_html=True)
                    
                    if not freq_df.empty:
                        size_counts = freq_df['Size'].value_counts().sort_index()
                        
                        fig3 = px.pie(
                            values=size_counts.values,
                            names=[f"{size} item" for size in size_counts.index],
                            color_discrete_sequence=px.colors.sequential.RdBu
                        )
                        fig3.update_layout(height=400)
                        st.plotly_chart(fig3, use_container_width=True)
    
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {str(e)}")
        st.info("💡 Pastikan format CSV sesuai dan coba lagi")

else:
    # Welcome screen
    st.markdown("""
        <div class="info-card">
            <h2 style="text-align: center; color: #00bcd4;"><i class="fas fa-hand-wave"></i> Selamat Datang!</h2>
            <p style="text-align: center; font-size: 1.1rem;">
                Aplikasi ini membantu Anda menganalisis pola pembelian produk menggunakan <strong>Algoritma Apriori</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="info-card">
                <h3 style="color:#000000;">
                    <i class="fass fa-bullseye"></i> Fitur Utama
                </h3>
                <ul style="color:#000000;">
                    <li><i class="fas fa-upload"></i> Upload data transaksi CSV</li>
                    <li><i class="fas fa-cog"></i> Atur parameter support & confidence</li>
                    <li><i class="fas fa-chart-bar"></i> Lihat frequent itemsets</li>
                    <li><i class="fas fa-link"></i> Analisis association rules</li>
                    <li><i class="fas fa-chart-line"></i> Visualisasi interaktif</li>
                    <li><i class="fas fa-download"></i> Download hasil analisis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card">
                <h3 style="color:#000000;">
                    <i class="fas fa-clipboard-list"></i> Format DATA CSV
                </h3>
                <p style="color:#000000;">File CSV harus memiliki 2 kolom:</p>
                <ol style="color:#000000;">
                    <li><strong>TransactionID</strong>: ID unik transaksi</li>
                    <li><strong>Items</strong>: Daftar produk (dipisah koma)</li>
                </ol>
                <p><strong>Contoh:</strong></p>
                <code>
                TransactionID,Items<br>
                T001,"Chitato, Oreo, Teh Botol"<br>
                T002,"Taro, Pocky, Aqua"
                </code>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-card" style="background: linear-gradient(135deg, #00bcd4 0%, #0097a7 100%); color: white;">
            <h3 style="color: white;"><i class="fas fa-rocket"></i> Mulai Sekarang!</h3>
            <p>Upload file CSV Anda di sidebar untuk memulai analisis</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p><i class="fas fa-chart-bar"></i> Market Basket Analysis dengan Algoritma Apriori</p>
        <p style="font-size: 0.9rem;">Dibuat untuk analisis pola pembelian minimarket</p>
    </div>
""", unsafe_allow_html=True)
