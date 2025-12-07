import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from config import PPOConfig, EXPERIMENT_PRESETS
from trainer import train_session


# 0. 유틸리티 함수: 결과 저장
def save_results_locally(experiment_name, score_history):
    """
    실험 결과를 로컬 'results' 폴더에 저장합니다.
    - CSV: 수치 데이터
    - PNG: 학습 곡선 그래프
    """
    # 폴더 생성
    save_dir = "results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = experiment_name.replace(" ", "_").replace(".", "")
    filename_base = f"{save_dir}/{safe_name}_{timestamp}"

    # 1. CSV 저장
    df = pd.DataFrame(score_history, columns=["Average Reward"])
    df.index.name = "Episode_x20"  # 20 에피소드 단위
    csv_path = f"{filename_base}.csv"
    df.to_csv(csv_path)

    # 2. 그래프 이미지 저장
    plt.figure(figsize=(10, 6))
    plt.plot(score_history, label=experiment_name)
    plt.title(f"Learning Curve: {experiment_name}")
    plt.xlabel("Index (x20 Episodes)")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    png_path = f"{filename_base}.png"
    plt.savefig(png_path)
    plt.close()  # 메모리 해제

    return csv_path, png_path



# 1. Streamlit 페이지 및 세션 초기화
st.set_page_config(page_title="RL PPO Dashboard", layout="wide")
st.title("PPO 학습 현황 대시보드")

# 세션 상태 초기화 (데이터 값은 유지)
if 'all_results' not in st.session_state:
    st.session_state['all_results'] = {}  # {실험이름: [점수리스트], ...}
if 'last_run_name' not in st.session_state:
    st.session_state['last_run_name'] = None

# 2. 사이드바: 실험 설정
st.sidebar.header("⚙️ 실험 설정")
preset_name = st.sidebar.selectbox("실험 모드 (Preset)", ["Custom"] + list(EXPERIMENT_PRESETS.keys()))

config = PPOConfig()
if preset_name != "Custom":
    preset = EXPERIMENT_PRESETS[preset_name]
    config.lr = preset['lr']
    config.eps_clip = preset['eps_clip']
    config.entropy_coef = preset['entropy_coef']
    st.sidebar.info(f"💡 {preset['description']}")

# 미세 조정
config.lr = st.sidebar.slider("Learning Rate", 0.0001, 0.01, config.lr, format="%.4f")
config.eps_clip = st.sidebar.slider("Clipping Epsilon", 0.01, 0.5, config.eps_clip)
config.entropy_coef = st.sidebar.slider("Entropy Coefficient", 0.0, 0.1, config.entropy_coef)
config.max_episodes = st.sidebar.number_input("최대 에피소드", 100, 5000, config.max_episodes)

# 3. 메인 화면: 탭 구성
tab1, tab2 = st.tabs(["🧪 실험 수행 (Experiment)", "📊 결과 비교 (Comparison)"])

# --- Tab 1: 실험 수행 ---
with tab1:
    st.subheader(f"Current Experiment: {preset_name}")

    # 학습 시작 버튼
    if st.button("🔥 학습 시작 (Start Training)"):
        # UI 컨테이너
        col1, col2 = st.columns(2)
        metric_epi = col1.empty()
        metric_score = col2.empty()
        chart_placeholder = st.empty()
        progress_bar = st.progress(0)

        score_history = []

        # 학습 루프 실행
        for n_epi, avg_score in train_session(config):
            score_history.append(avg_score)

            # 실시간 UI 업데이트
            metric_epi.metric("Episode", n_epi)
            metric_score.metric("Avg Score (Last 20)", f"{avg_score:.1f}")
            chart_placeholder.line_chart(score_history)
            progress = min(n_epi / config.max_episodes, 1.0)
            progress_bar.progress(progress)

            if avg_score > 200:
                st.success(f"🎉 Solved! Episode {n_epi}")
                break

        # 학습 완료 후 세션에 데이터 저장
        st.session_state['all_results'][preset_name] = score_history
        st.session_state['last_run_name'] = preset_name
        st.success("학습 종료! 데이터가 저장되었습니다.")

    # [기능 1] 결과 저장 버튼 (학습 직후 표시)
    if st.session_state['last_run_name'] is not None:
        last_name = st.session_state['last_run_name']
        last_scores = st.session_state['all_results'].get(last_name, [])

        st.divider()
        st.write(f"📂 **'{last_name}'** 실험 결과 관리")

        if st.button("💾 결과 로컬 저장 (Save CSV & Image)"):
            csv_path, png_path = save_results_locally(last_name, last_scores)
            st.success(f"저장 완료!\n- 데이터: {csv_path}\n- 이미지: {png_path}")

# --- Tab 2: 결과 비교 ---
with tab2:
    st.subheader("📈 다중 실험 결과 비교 분석")

    results = st.session_state['all_results']

    if not results:
        st.info("아직 수행된 실험이 없습니다. '실험 수행' 탭에서 학습을 진행해주세요.")
    else:
        # [기능 2] 겹쳐서 그리기 코드
        st.markdown(f"총 **{len(results)}** 건의 실험 결과가 있습니다.")

        # Matplotlib을 사용하여 겹쳐 그리기
        fig, ax = plt.subplots(figsize=(10, 6))

        for exp_name, scores in results.items():
            ax.plot(scores, label=exp_name, alpha=0.8, linewidth=2)

        ax.set_title("Learning Curve Comparison")
        ax.set_xlabel("Steps (x20 Episodes)")
        ax.set_ylabel("Average Reward")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        # Streamlit에 그래프 표시
        st.pyplot(fig)

        # 데이터 테이블 표시 (옵션)
        with st.expander("상세 데이터 보기"):
            # 길이가 다를 수 있으므로 DataFrame 생성 시 유의
            df_compare = pd.DataFrame({k: pd.Series(v) for k, v in results.items()})
            st.dataframe(df_compare)