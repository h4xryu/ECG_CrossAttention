#!/usr/bin/env python3
"""
analyze_results.py - 실험 결과 비교 및 시각화 스크립트

사용법:
    python analyze_results.py                      # ECG_Results 폴더 전체 스캔
    python analyze_results.py --exp_dir ./ECG_Results/experiment_xxx  # 특정 실험만
    
기능:
    1. 여러 실험의 결과 CSV를 읽어서 비교 테이블 생성
    2. 성능 지표 그래프화 (Bar chart)
    3. 클래스별 성능 비교
"""

import os
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 설정
# =============================================================================
RESULTS_DIR = "./ECG_Results"
OUTPUT_DIR = "./ECG_Results/comparison"

# 관심 있는 주요 지표
KEY_METRICS = ['Accuracy', 'Macro_F1', 'Weighted_F1', 'S_Recall', 'S_F1']


def find_result_csvs(base_dir):
    """실험 폴더에서 결과 CSV 파일들을 찾음"""
    pattern = os.path.join(base_dir, "**/results_*.csv")
    csvs = glob.glob(pattern, recursive=True)
    return sorted(csvs)


def parse_csv_to_summary(csv_path):
    """
    CSV 파일을 읽어서 핵심 지표 추출
    Returns: dict with experiment name and metrics
    """
    df = pd.read_csv(csv_path)
    
    # 실험 이름 추출 (폴더명에서)
    exp_folder = os.path.dirname(csv_path)
    exp_name = os.path.basename(exp_folder)
    if "experiment_" in exp_name:
        # experiment_20260115_143000_baseline -> baseline
        parts = exp_name.split("_")
        if len(parts) >= 4:
            exp_name = "_".join(parts[3:])
    
    # 모델 타입 추출 (파일명에서)
    csv_name = os.path.basename(csv_path)
    model_type = csv_name.replace("results_", "").replace(".csv", "")
    
    summary = {
        'exp_name': exp_name,
        'model_type': model_type,
        'csv_path': csv_path,
    }
    
    # Overall Accuracy
    acc_row = df[df['Metric_Type'] == 'Overall']
    if not acc_row.empty:
        summary['Accuracy'] = acc_row['Precision'].values[0]
    
    # Macro 지표
    macro_row = df[df['Metric_Type'] == 'Macro_Avg']
    if not macro_row.empty:
        summary['Macro_Precision'] = macro_row['Precision'].values[0]
        summary['Macro_Recall'] = macro_row['Recall'].values[0]
        summary['Macro_F1'] = macro_row['F1-Score'].values[0]
    
    # Weighted 지표
    weighted_row = df[df['Metric_Type'] == 'Weighted_Avg']
    if not weighted_row.empty:
        summary['Weighted_Precision'] = weighted_row['Precision'].values[0]
        summary['Weighted_Recall'] = weighted_row['Recall'].values[0]
        summary['Weighted_F1'] = weighted_row['F1-Score'].values[0]
    
    # 클래스별 지표
    for cls in ['N', 'S', 'V', 'F']:
        cls_row = df[(df['Metric_Type'] == 'Per_Class') & (df['Class'] == cls)]
        if not cls_row.empty:
            summary[f'{cls}_Precision'] = cls_row['Precision'].values[0]
            summary[f'{cls}_Recall'] = cls_row['Recall'].values[0]
            summary[f'{cls}_F1'] = cls_row['F1-Score'].values[0]
    
    return summary


def create_comparison_table(summaries):
    """여러 실험 결과를 비교 테이블로 정리"""
    df = pd.DataFrame(summaries)
    
    # 실험명 + 모델타입을 합쳐서 고유 식별자로
    df['Experiment'] = df['exp_name'] + ' (' + df['model_type'] + ')'
    
    # 컬럼 정렬
    cols = ['Experiment', 'Accuracy', 
            'Macro_F1', 'Weighted_F1',
            'S_Recall', 'S_F1', 'V_Recall', 'V_F1']
    cols = [c for c in cols if c in df.columns]
    
    return df[cols].round(4)


def plot_comparison_bar(df, metrics, save_path):
    """
    여러 실험의 지표를 Bar chart로 비교
    """
    # 데이터 준비
    plot_data = df[['Experiment'] + metrics].melt(
        id_vars='Experiment', 
        var_name='Metric', 
        value_name='Score'
    )
    
    # 플롯 생성
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=plot_data, x='Metric', y='Score', hue='Experiment')
    
    plt.title('Experiment Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Score')
    plt.xlabel('')
    plt.ylim(0, 1.0)
    
    # 범례 위치 조정
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {save_path}")


def plot_class_performance(df, save_path):
    """
    클래스별 F1 성능 히트맵
    """
    # 클래스별 F1 추출
    f1_cols = ['N_F1', 'S_F1', 'V_F1', 'F_F1']
    f1_cols = [c for c in f1_cols if c in df.columns]
    
    if not f1_cols:
        return
    
    heatmap_data = df[['Experiment'] + f1_cols].set_index('Experiment')
    heatmap_data.columns = [c.replace('_F1', '') for c in heatmap_data.columns]
    
    plt.figure(figsize=(8, max(4, len(df) * 0.5)))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, linewidths=0.5)
    
    plt.title('Class-wise F1 Score', fontsize=14, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Experiment')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {save_path}")


def print_summary_table(df):
    """결과 테이블을 보기 좋게 출력"""
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*100)
    
    # pandas 출력 설정
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    print(df.to_string(index=False))
    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(description='실험 결과 비교 및 시각화')
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='특정 실험 폴더 (미지정시 전체 스캔)')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='출력 폴더')
    args = parser.parse_args()
    
    # CSV 파일 찾기
    if args.exp_dir:
        search_dir = args.exp_dir
    else:
        search_dir = RESULTS_DIR
    
    csv_files = find_result_csvs(search_dir)
    
    if not csv_files:
        print(f"No result CSV files found in {search_dir}")
        return
    
    print(f"Found {len(csv_files)} result CSV files:")
    for f in csv_files:
        print(f"  - {f}")
    
    # 결과 요약 추출
    summaries = []
    for csv_path in csv_files:
        try:
            summary = parse_csv_to_summary(csv_path)
            summaries.append(summary)
        except Exception as e:
            print(f"Error parsing {csv_path}: {e}")
    
    if not summaries:
        print("No valid results to compare.")
        return
    
    # 비교 테이블 생성
    df = create_comparison_table(summaries)
    print_summary_table(df)
    
    # 출력 폴더 생성
    os.makedirs(args.output, exist_ok=True)
    
    # CSV로 저장
    csv_output = os.path.join(args.output, 'comparison_summary.csv')
    df.to_csv(csv_output, index=False)
    print(f"Summary CSV saved: {csv_output}")
    
    # 차트 생성
    if len(df) >= 1:
        # 주요 지표 비교 바차트
        metrics_to_plot = [m for m in KEY_METRICS if m in df.columns]
        if metrics_to_plot:
            bar_path = os.path.join(args.output, 'comparison_bar.png')
            plot_comparison_bar(df, metrics_to_plot, bar_path)
        
        # 클래스별 F1 히트맵
        heatmap_path = os.path.join(args.output, 'class_f1_heatmap.png')
        plot_class_performance(df, heatmap_path)
    
    print(f"\nAll outputs saved to: {args.output}")


if __name__ == "__main__":
    main()

