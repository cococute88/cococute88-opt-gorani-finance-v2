TOSS_CSS = """
<style>
.toss-card { background: #ffffff; padding: 24px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.04); margin-bottom: 24px; }
.toss-metric { margin-bottom: 8px; }
.toss-metric .label { font-size: 14px; color: #6b7684; font-weight: 500; margin-bottom: 4px; }
.toss-metric .value { font-size: 22px; font-weight: 700; color: #191f28; }
.toss-metric .sub { font-size: 13px; color: #8b95a1; margin-top: 2px; }
.toss-metric.accent .value { color: #3182F6; }
.toss-metric.success .value { color: #00875A; }
.toss-metric.warn .value { color: #FF8B00; }
.toss-note { background: #f2f4f6; padding: 16px; border-radius: 12px; font-size: 14px; color: #4e5968; margin-bottom: 20px; line-height: 1.5; }
.toss-badge { display: inline-block; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; }
.badge-active { background: #e8f3ff; color: #1b64da; }
.badge-retire { background: #fff0d6; color: #cc7000; }
.badge-withdraw { background: #feecea; color: #d93d44; }
.badge-info { background: #f2f4f6; color: #4e5968; }
.cloud-status { padding: 10px 14px; border-radius: 8px; font-weight: 600; font-size: 13px; margin-top: 10px; }
.cloud-connected { background: #e5f6ed; color: #00875a; }
.cloud-disconnected { background: #feecea; color: #d93d44; }
.cloud-testing { background: #fff0d6; color: #cc7000; }
/* 모바일 환경 타이틀 및 서브타이틀 일괄 축소 */
@media screen and (max-width: 768px) {
    /* 1. 메인 타이틀 (h1) */
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stHeadingWithActionElements"] h1 {
        font-size: 24px !important; 
        white-space: nowrap !important; 
        word-break: keep-all !important;
        letter-spacing: -1px !important; 
    }
    
    /* 2. 서브 타이틀 (h2, h3) - 기본설정, 데이터 입력 등 */
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stHeadingWithActionElements"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stHeadingWithActionElements"] h3 {
        font-size: 18px !important; /* 메인 타이틀(24px)보다 작게 고정 */
        word-break: keep-all !important;
        letter-spacing: -0.5px !important;
        margin-top: 1rem !important; /* 위쪽 여백 살짝 줄임 */
        margin-bottom: 0.5rem !important;
    }
}
</style>
"""
