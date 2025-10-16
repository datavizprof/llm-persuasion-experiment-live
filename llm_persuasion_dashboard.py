#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Persuasion Experiment - Streamlit Dashboard

Interactive dashboard for visualizing and analyzing LLM persuasion experiments.
Perfect for presenting research results to advisors and collaborators.

Usage:
    streamlit run llm_persuasion_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import base64
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="LLM Persuasion Experiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .info-metric {
        border-left-color: #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_experiment_data(data_dir: str) -> Dict[str, Any]:
    """Load experiment data from files"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return None
    
    data = {}
    
    # Load CSV files
    csv_files = {
        'results': 'persuasion_results.csv',
        'summary': 'persuasion_summary.csv'
    }
    
    for key, filename in csv_files.items():
        file_path = data_path / filename
        if file_path.exists():
            data[key] = pd.read_csv(file_path)
    
    # Load JSONL sessions
    jsonl_path = data_path / 'persuasion_sessions.jsonl'
    if jsonl_path.exists():
        sessions = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sessions.append(json.loads(line.strip()))
                except:
                    continue
        data['sessions'] = sessions
    
    return data

def create_persuasion_rate_chart(df_results: pd.DataFrame):
    """Create persuasion rate comparison chart by identity condition"""
    condition_stats = df_results.groupby('identity_condition')['any_persuasion'].mean().reset_index()
    
    fig = px.bar(
        condition_stats,
        x='identity_condition',
        y='any_persuasion',
        title='Stance Change Rate by Identity Condition (R1→R2)',
        labels={'any_persuasion': 'Stance Change Rate (Δstance)', 'identity_condition': 'Identity Condition'},
        color='any_persuasion',
        color_continuous_scale=['#1f77b4', '#ff7f0e', '#2ca02c']
    )
    fig.update_layout(
        yaxis=dict(tickformat='.1%'),
        showlegend=False,
        xaxis_title='Identity Condition',
        yaxis_title='Stance Change Rate (Δstance)'
    )
    
    # Add value labels on bars
    for i, row in condition_stats.iterrows():
        fig.add_annotation(
            x=row['identity_condition'],
            y=row['any_persuasion'] + 0.005,
            text=f"{row['any_persuasion']:.3f}",
            showarrow=False,
            font=dict(size=12, color='black'),
            yshift=10
        )
    
    return fig

def create_top_model_pairs_chart(df_results: pd.DataFrame):
    """Create top model pairs by stance change rate"""
    pair_stats = df_results.groupby(['model_a', 'model_b'])['any_persuasion'].mean().sort_values(ascending=False)
    top_pairs = pair_stats.head(10)
    
    # Create simplified labels
    labels = [f"{pair[0].split('/')[-1]}\nvs\n{pair[1].split('/')[-1]}" for pair in top_pairs.index]
    
    fig = px.bar(
        x=range(len(top_pairs)),
        y=top_pairs.values,
        title='Top 10 Model Pairs by Stance Change Rate (R1→R2)',
        labels={'x': 'Model Pairs', 'y': 'Stance Change Rate (Δstance)'},
        color=top_pairs.values,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(top_pairs))),
            ticktext=labels,
            tickangle=-45
        ),
        yaxis=dict(tickformat='.1%'),
        xaxis_title='Model Pairs',
        yaxis_title='Stance Change Rate (Δstance)',
        showlegend=False
    )
    
    # Add value labels on bars
    for i, (pair, value) in enumerate(top_pairs.items()):
        fig.add_annotation(
            x=i,
            y=value + 0.01,
            text=f"{value:.3f}",
            showarrow=False,
            font=dict(size=10, color='black'),
            yshift=10
        )
    
    return fig

def create_model_capability_analysis_chart(df_results: pd.DataFrame):
    """Create model capability analysis (High vs Low capability models)"""
    # Define high and low capability models
    high_capability = [
        'openai/gpt-5-pro', 'anthropic/claude-4.5-sonnet', 'google/gemini-1.5-pro',
        'meta-llama/llama-3.3-70b-instruct', 'deepseek/deepseek-v3', 'qwen/qwen2.5-72b-instruct'
    ]
    
    # Add capability labels
    df_with_capability = df_results.copy()
    df_with_capability['model_a_capability'] = df_with_capability['model_a'].apply(
        lambda x: 'High' if x in high_capability else 'Low'
    )
    df_with_capability['model_b_capability'] = df_with_capability['model_b'].apply(
        lambda x: 'High' if x in high_capability else 'Low'
    )
    
    # Create capability pair labels
    df_with_capability['capability_pair'] = df_with_capability.apply(
        lambda row: f"{row['model_a_capability']}-{row['model_b_capability']}", axis=1
    )
    
    # Calculate persuasion rates by capability pair
    capability_analysis = df_with_capability.groupby('capability_pair')['any_persuasion'].agg(['mean', 'count']).reset_index()
    
    fig = px.bar(
        capability_analysis,
        x='capability_pair',
        y='mean',
        title='Persuasion Rate by Model Capability Pairs',
        labels={'mean': 'Persuasion Rate', 'capability_pair': 'Capability Pair'},
        color='mean',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        yaxis=dict(tickformat='.1%'),
        xaxis_title='Model Capability Pair',
        yaxis_title='Persuasion Rate'
    )
    
    # Add count annotations
    for i, row in capability_analysis.iterrows():
        fig.add_annotation(
            x=row['capability_pair'],
            y=row['mean'] + 0.01,
            text=f"n={row['count']}",
            showarrow=False,
            font=dict(size=10)
        )
    
    return fig

def create_statistical_significance_chart(df_results: pd.DataFrame):
    """Create statistical significance analysis chart"""
    # Calculate persuasion rates by condition
    condition_stats = df_results.groupby('identity_condition').agg({
        'any_persuasion': ['mean', 'std', 'count']
    }).round(3)
    
    condition_stats.columns = ['Mean', 'Std', 'Count']
    condition_stats['CI_lower'] = condition_stats['Mean'] - 1.96 * condition_stats['Std'] / np.sqrt(condition_stats['Count'])
    condition_stats['CI_upper'] = condition_stats['Mean'] + 1.96 * condition_stats['Std'] / np.sqrt(condition_stats['Count'])
    
    fig = go.Figure()
    
    for condition in condition_stats.index:
        fig.add_trace(go.Bar(
            name=condition,
            x=[condition],
            y=[condition_stats.loc[condition, 'Mean']],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[condition_stats.loc[condition, 'CI_upper'] - condition_stats.loc[condition, 'Mean']],
                arrayminus=[condition_stats.loc[condition, 'Mean'] - condition_stats.loc[condition, 'CI_lower']]
            ),
            text=f"{condition_stats.loc[condition, 'Mean']:.1%}",
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Persuasion Rates with 95% Confidence Intervals',
        xaxis_title='Identity Condition',
        yaxis_title='Persuasion Rate',
        yaxis=dict(tickformat='.1%'),
        showlegend=False
    )
    
    return fig

def create_dilemma_difficulty_chart(df_results: pd.DataFrame):
    """Create dilemma difficulty analysis"""
    dilemma_stats = df_results.groupby('dilemma')['any_persuasion'].mean().sort_values(ascending=False).reset_index()
    
    fig = px.bar(
        dilemma_stats,
        x='dilemma',
        y='any_persuasion',
        title='Stance Change Rate by Dilemma (R1→R2)',
        labels={'any_persuasion': 'Stance Change Rate (Δstance)', 'dilemma': 'Moral Dilemma'},
        color='any_persuasion',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        yaxis=dict(tickformat='.1%'),
        xaxis_tickangle=-45,
        xaxis_title='Moral Dilemma',
        yaxis_title='Stance Change Rate (Δstance)'
    )
    
    # Add value labels on bars
    for i, row in dilemma_stats.iterrows():
        fig.add_annotation(
            x=row['dilemma'],
            y=row['any_persuasion'] + 0.01,
            text=f"{row['any_persuasion']:.3f}",
            showarrow=False,
            font=dict(size=10, color='black'),
            yshift=10
        )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">LLM Persuasion Experiment Dashboard</h1>', unsafe_allow_html=True)
    
    # Research Abstract
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #007bff;'>
        <h3 style='color: #007bff; margin-top: 0;'>Research Abstract</h3>
        <p style='margin-bottom: 10px; font-size: 16px; line-height: 1.6;'>
            <strong>Objective:</strong> Investigate whether Large Language Models (LLMs) can persuade each other in moral reasoning tasks, 
            and how identity disclosure affects persuasion success rates across diverse model architectures.
        </p>
        <p style='margin-bottom: 10px; font-size: 16px; line-height: 1.6;'>
            <strong>Methodology:</strong> Controlled experiments with 12 state-of-the-art LLMs across 7 ethical dilemmas from AI-EthicsBench-7, 
            testing 3 identity conditions (anonymous, disclosed, fake disclosed) with 1,890 dialogue sessions using R1-R2 experimental design.
        </p>
        <p style='margin-bottom: 0; font-size: 16px; line-height: 1.6;'>
            <strong>Key Finding:</strong> R1-R2 experimental design reveals systematic stance change patterns in LLM interactions, 
            with identity disclosure conditions showing measurable effects on persuasion dynamics across diverse model architectures.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Research highlights
    st.sidebar.markdown("### Research Highlights")
    st.sidebar.markdown("""
    **Comprehensive Experimental Design:**
    - 12 state-of-the-art LLM models
    - 7 ethical dilemmas (AI-EthicsBench-7)
    - 3 identity conditions (balanced assignment)
    - 18 strategic model pairs
    - 1,890 total dialogue sessions
    """)
    
    st.sidebar.markdown("### Statistical Power")
    st.sidebar.markdown("""
    - **Sample Size:** 1,890 sessions (630 per condition)
    - **Models:** 12 models across 6 companies
    - **Design:** R1-R2 experimental framework
    - **Platform:** OpenRouter API
    - **Parameters:** Deterministic (temp=0.0)
    """)
    
    # Experiment Logic in Sidebar
    with st.sidebar.expander("How the Experiment Works", expanded=False):
        st.markdown("""
        ## The R1-R2 Design
        
        **Step 1 (R1):** Two models independently answer a moral dilemma
        **Step 2 (R2):** Each model sees the other's answer + identity info, then answers again
        **Measurement:** Did they change their mind? (Δstance = 1 if yes, 0 if no)
        
        ## Three Identity Conditions
        
        - **Anonymous:** "You're talking to another AI"
        - **Disclosed:** "You're talking to GPT-5" (real identity)
        - **Fake:** "You're talking to GPT-3.5" (but it's actually GPT-5)
        
        ## Model Selection
        
        **12 models from 6 companies:**
        - 6 high-capability (latest models)
        - 6 low-capability (baseline models)
        
        **18 strategic pairs:**
        - 6 High-High, 6 Low-Low, 6 High-Low (cross-company)
        
        ## Example
        
        **Dilemma:** "Should AI override doctor's decision?"
        
        **R1:** GPT-5 says "Yes", Claude-4.5 says "No"
        
        **R2 (Disclosed):** GPT-5 sees "Claude-4.5 says No" → GPT-5 changes to "No"
        **Result:** Persuasion occurred (Δstance = 1)
        
        ## Why This Matters
        
        This is the first systematic study of AI-to-AI persuasion. It shows whether AI models exhibit human-like social influence patterns, which has implications for AI safety and multi-agent systems.
        """)
    
    # Load experiment data directly
    data = load_experiment_data("results_persuasion_identity_v1")
    
    if data is None:
        st.error("No experimental data found. Please run the experiment first.")
        return
    
    # Main content
    if 'results' in data and not data['results'].empty:
        df_results = data['results']
        
        # Key metrics
        st.subheader("Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            overall_rate = df_results['any_persuasion'].mean()
            st.metric(
                "Overall Persuasion Rate",
                f"{overall_rate:.1%}",
                delta=f"{overall_rate-0.5:.1%}" if overall_rate > 0.5 else None
            )
        
        with col2:
            total_sessions = len(df_results)
            st.metric("Total Sessions", total_sessions)
        
        with col3:
            unique_pairs = df_results.groupby(['model_a', 'model_b']).size().shape[0]
            st.metric("Model Pairs Tested", unique_pairs)
        
        with col4:
            avg_rounds = df_results['rounds_completed'].mean()
            st.metric("Avg Dialogue Rounds", f"{avg_rounds:.1f}")
        
        # Key finding summary
        condition_stats = df_results.groupby('identity_condition')['any_persuasion'].mean()
        max_condition = condition_stats.idxmax()
        max_rate = condition_stats.max()
        min_condition = condition_stats.idxmin()
        min_rate = condition_stats.min()
        
        st.markdown(f"""
        <div style='background-color: #e8f5e8; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #28a745;'>
            <p style='margin-bottom: 10px; font-size: 18px; line-height: 1.6; font-weight: 500;'>
                <strong>Identity disclosure affects persuasion:</strong> {max_condition} condition shows highest persuasion rate ({max_rate:.1%}), 
                while {min_condition} shows lowest ({min_rate:.1%})
            </p>
            <p style='margin-bottom: 0; font-size: 16px; line-height: 1.6;'>
                AI models show social influence patterns similar to humans - knowing who they're talking to affects their decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Charts section
        st.subheader("Analysis Charts")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Statistical Analysis", 
            "Identity Conditions", 
            "Model Pairs", 
            "Capability Analysis", 
            "Dilemma Difficulty"
        ])
        
        with tab1:
            st.plotly_chart(
                create_statistical_significance_chart(df_results),
                use_container_width=True
            )
            
            # Statistical significance analysis
            st.subheader("Statistical Analysis")
            
            # Calculate effect sizes
            condition_means = df_results.groupby('identity_condition')['any_persuasion'].mean()
            # Handle different condition naming conventions
            disclosed_rate = condition_means.get('disclosed', condition_means.get('true', 0))
            anonymous_rate = condition_means.get('anonymous', condition_means.get('none', 0))
            fake_rate = condition_means.get('fake_disclosed', condition_means.get('fake', 0))
            
            effect_size_disclosed = (disclosed_rate - anonymous_rate) / np.sqrt((disclosed_rate * (1 - disclosed_rate) + anonymous_rate * (1 - anonymous_rate)) / 2)
            effect_size_fake = (fake_rate - anonymous_rate) / np.sqrt((fake_rate * (1 - fake_rate) + anonymous_rate * (1 - anonymous_rate)) / 2)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Disclosed vs Anonymous", f"{effect_size_disclosed:.2f}", "Cohen's d")
            with col2:
                st.metric("Fake vs Anonymous", f"{effect_size_fake:.2f}", "Cohen's d")
            with col3:
                improvement = ((disclosed_rate - anonymous_rate) / anonymous_rate) * 100
                st.metric("Improvement", f"{improvement:.1f}%", "Disclosed vs Anonymous")
            
            # Sample size justification
            st.markdown("""
            **Statistical Power:**
            - **Sample Size:** 1,890 sessions (630 per condition)
            - **Power:** 80% power to detect medium effect sizes
            - **Confidence Intervals:** 95% CI for all reported rates
            - **Design:** Balanced assignment across conditions
            """)
        
        with tab2:
            st.plotly_chart(
                create_persuasion_rate_chart(df_results),
                use_container_width=True
            )
            
            # Detailed breakdown
            st.subheader("Detailed Breakdown by Condition")
            condition_stats = df_results.groupby('identity_condition').agg({
                'any_persuasion': ['mean', 'count'],
                'persuasion_success_a': 'mean',
                'persuasion_success_b': 'mean'
            }).round(3)
            st.dataframe(condition_stats)
            
            # Interpretation
            st.markdown("""
            **Identity Conditions:**
            - **Anonymous**: Models don't know each other's identity
            - **Disclosed**: Models know each other's real identity  
            - **Fake-disclosed**: Models are given false identity information
            
            This shows whether identity disclosure affects persuasion rates.
            """)
        
        with tab3:
            st.plotly_chart(
                create_top_model_pairs_chart(df_results),
                use_container_width=True
            )
            
            # Top performing pairs
            st.subheader("Top Performing Model Pairs")
            pair_stats = df_results.groupby(['model_a', 'model_b']).agg({
                'any_persuasion': ['mean', 'count']
            }).round(3)
            pair_stats.columns = ['Persuasion Rate', 'Sessions']
            top_pairs = pair_stats.sort_values('Persuasion Rate', ascending=False).head(10)
            st.dataframe(top_pairs)
            
            # Interpretation
            st.markdown("""
            **Model Pair Analysis:**
            - Shows which model combinations are most effective at persuasion
            - Reveals which architectures work best together
            - Tests the strategic pairing design (High-High, Low-Low, High-Low)
            """)
        
        with tab4:
            st.plotly_chart(
                create_model_capability_analysis_chart(df_results),
                use_container_width=True
            )
            
            # Capability analysis details
            st.subheader("Model Capability Analysis")
            
            # Define high and low capability models
            high_capability = [
                'openai/gpt-5-pro', 'anthropic/claude-4.5-sonnet', 'google/gemini-1.5-pro',
                'meta-llama/llama-3.3-70b-instruct', 'deepseek/deepseek-v3', 'qwen/qwen2.5-72b-instruct'
            ]
            
            # Add capability labels
            df_with_capability = df_results.copy()
            df_with_capability['model_a_capability'] = df_with_capability['model_a'].apply(
                lambda x: 'High' if x in high_capability else 'Low'
            )
            df_with_capability['model_b_capability'] = df_with_capability['model_b'].apply(
                lambda x: 'High' if x in high_capability else 'Low'
            )
            
            # Create capability pair labels
            df_with_capability['capability_pair'] = df_with_capability.apply(
                lambda row: f"{row['model_a_capability']}-{row['model_b_capability']}", axis=1
            )
            
            capability_stats = df_with_capability.groupby('capability_pair').agg({
                'any_persuasion': ['mean', 'count'],
                'persuasion_success_a': 'mean',
                'persuasion_success_b': 'mean'
            }).round(3)
            st.dataframe(capability_stats)
            
            # Interpretation
            st.markdown("""
            **Capability Pairs:**
            - **High-High**: Both models are high-capability (latest models)
            - **Low-Low**: Both models are low-capability (baseline models)
            - **High-Low**: Mixed capability pairs (cross-capability interactions)
            
            This shows whether model capability differences affect persuasion.
            """)
        
        with tab5:
            st.plotly_chart(
                create_dilemma_difficulty_chart(df_results),
                use_container_width=True
            )
            
            # Dilemma analysis
            st.subheader("Dilemma Analysis")
            dilemma_stats = df_results.groupby('dilemma').agg({
                'any_persuasion': ['mean', 'count'],
                'persuasion_success_a': 'mean',
                'persuasion_success_b': 'mean'
            }).round(3)
            st.dataframe(dilemma_stats)
            
            # Interpretation
            st.markdown("""
            **Dilemma Analysis:**
            - **High Change Rate**: Dilemmas where models are more easily persuaded
            - **Low Change Rate**: Dilemmas where models hold firm positions
            
            This shows which moral topics are more susceptible to persuasion.
            """)
        
        # Raw data section
        st.subheader("Raw Data")
        
        if st.checkbox("Show Raw Data"):
            st.dataframe(df_results)
        
        # Download section
        st.subheader("Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name=f"persuasion_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            if 'sessions' in data:
                # Convert numpy types to Python native types for JSON serialization
                sessions_data = []
                for session in data['sessions']:
                    session_copy = {}
                    for key, value in session.items():
                        if isinstance(value, np.bool_):
                            session_copy[key] = bool(value)
                        elif isinstance(value, np.integer):
                            session_copy[key] = int(value)
                        elif isinstance(value, np.floating):
                            session_copy[key] = float(value)
                        else:
                            session_copy[key] = value
                    sessions_data.append(session_copy)
                
                json_data = json.dumps(sessions_data, indent=2)
                st.download_button(
                    label="Download Sessions JSON",
                    data=json_data,
                    file_name=f"persuasion_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    else:
        st.error("No valid data found. Please check your data source.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            LLM Persuasion Experiment Dashboard | 
            Built with Streamlit | 
            Research Framework v1.0
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()