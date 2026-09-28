#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# submit_lib.sh  —  submit.sh 与 pilot/submit_pilot.sh 共用的函数（被 source，不单独运行）。
# 纯 bash，不调 python（登录节点可跑）。
#
# 1. run 的状态（D-053）：done / stale / todo
#      done  = metrics.json 存在、exit_code 为 0、config_sha 等于 INDEX
#      stale = exit_code 为 0 但 config_sha 不等（配置在结果回来之后改过；与 status.py 同名）
#              → 默认**不重交**，RESUBMIT_STALE=1 才交。已经判完的组（如 G7）不会因为
#                base.yaml 改了一个评估开关就被整组重跑。
#      todo  = 其余（没有文件、exit_code 非 0 —— 含 pack.sbatch 判出的 OOM）
#
# 2. 一卡多跑（D-048 / D-052），PACK=K>1 时生效；PACK 未设 / =1 → 调用方照旧一卡一跑。
#    核心：**不触发 OOM、尽量省机时**。
#      · 分桶：同一个包里只放**同一格子**（group__cell）的不同 seed —— 同一种配置类型。
#      · 探路：格子还没有任何 <group>__<cell>__pack-*.gpu.json → 只交一个 PROBE_K（缺省 2）的包，
#        其余打印 held，等显存数据回来后再跑一次本脚本。5 个 seed = 2（探路）+ 3，包数与 3+2 相同。
#      · 定 K：K_cell = min(PACK, ⌊mem_total × PACK_MEM_PCT% / (run_peak_max + PACK_CTX_MIB)⌋)。
#        run_peak 是 TF 分配器的真实峰值（[GPUMem]）；没有它就退回 mem_max / k（整卡读数，偏保守）；
#        连容量都没有 → 不超过已经跑通过的 k。PACK_MEM_PCT=85、PACK_CTX_MIB=1024 **没有证据**，
#        第一批满长包回来后按实测校准（DECISIONS D-052）。
#      · OOM：该格子任何一个包出过 OOM → K_cell ≤ 那个包的 k − 1。OOM 的 run 自动是 todo。
#      · 每个包：sbatch -c $((4*n)) --job-name=exp3v2-pack$n pack.sbatch <tag> <三元组…>，
#        tag = <group>__<cell>__pack-k<n>-s<首个 seed>（带 k：降档重交不会覆盖出事那个包的记录）。
#      · 跨格子合包（D-060，PACK_MIX=1 缺省开；=0 关）：**只处理余数** —— 每个格子按自己的 K 切完满包后
#        剩下的那一包（不满 K）。同一组里，满足下面全部条件的格子，余数合起来按 K_mix 切包：
#          - 该格子的 K 来自 TF 的**真实峰值**（不是整卡读数 / 跑通过的 k），且历史上没有 OOM、没有显存告警；
#          - 这些格子的真实峰值相差 ≤ PACK_MIX_TOL_PCT%（缺省 10%，按最大值算；不满足 → 整组都不合）；
#          - 合包后作业数**严格变少**（否则保持同格子，不为混而混）。
#        K_mix = 参与格子 K 的最小值。合包的 tag = <group>__mix-<cellA>+<cellB>+…__pack-k<n>-s<首个 seed>；
#        各格子定 K 时也读它参与过的合包的 gpu.json（峰值取整包最大、OOM 对包里每个格子都生效 —— 偏保守）。
#        起因：G6 每格 3 个 seed，探路包用掉 2 个，剩下 3 个格子各 1 个 → 不合就是 3 张卡各跑 1 个。
#
# 调用方要先设好：ROOT、PACK_JOB（pack.sbatch 的路径）、DRY、SUBMIT_OK（1 = 真提交）。
# **注意**：本脚本不知道 SLURM 队列里已有什么 —— 交完等回传再跑下一次，别连着跑两遍。
# ══════════════════════════════════════════════════════════════════════════

PACK="${PACK:-1}"
PROBE_K="${PROBE_K:-2}"
PACK_MEM_PCT="${PACK_MEM_PCT:-85}"
PACK_CTX_MIB="${PACK_CTX_MIB:-1024}"
RESUBMIT_STALE="${RESUBMIT_STALE:-0}"
PACK_MIX="${PACK_MIX:-1}"
PACK_MIX_TOL_PCT="${PACK_MIX_TOL_PCT:-10}"
for _v in PACK PROBE_K PACK_MEM_PCT PACK_CTX_MIB PACK_MIX_TOL_PCT; do
    if ! [[ "${!_v}" =~ ^[0-9]+$ ]] || [ "${!_v}" -lt 1 ]; then
        echo "[submit] $_v 必须是正整数，收到 '${!_v}'" >&2; exit 2
    fi
done
if [ "$PACK_MIX" != 0 ] && [ "$PACK_MIX" != 1 ]; then
    echo "[submit] PACK_MIX 只能是 0 或 1，收到 '$PACK_MIX'" >&2; exit 2
fi
held=0

# run_state <metrics 绝对路径> <INDEX 里的 config_sha>  →  打印 done / stale / todo
run_state() {
    local m="$1" sha="$2"
    if [ -s "$m" ] && grep -q '"exit_code": *0' "$m"; then
        if grep -q "\"config_sha\": *\"$sha\"" "$m"; then echo done; else echo stale; fi
    else
        echo todo
    fi
    return 0
}

# ── 分桶 ──────────────────────────────────────────────────────────────────
declare -A _PK_BUCKET=()        # key → 空格分隔的下标
_PK_KEYS=()                     # key 的出现顺序（与 INDEX 顺序一致）
_PK_CFG=(); _PK_RID=(); _PK_MET=(); _PK_MABS=(); _PK_SEED=(); _PK_CELL=()
# 余数跨格子合包（D-060）：key → 余数下标 / 该格子的 K / 真实峰值
declare -A _MIX_REM=() _MIX_K=() _MIX_PEAK=()
_MIX_KEYS=()

pack_add() {                    # pack_add <group> <cell> <seed> <config> <run_id> <metrics> <metrics_abs>
    local key="$1__$2" i=${#_PK_RID[@]}
    _PK_SEED+=("$3"); _PK_CFG+=("$4"); _PK_RID+=("$5"); _PK_MET+=("$6"); _PK_MABS+=("$7")
    _PK_CELL+=("$2")
    if [ -z "${_PK_BUCKET[$key]+x}" ]; then
        _PK_KEYS+=("$key"); _PK_BUCKET[$key]="$i"
    else
        _PK_BUCKET[$key]="${_PK_BUCKET[$key]} $i"
    fi
    return 0
}

# 从一个 gpu.json 里取标量（null / 缺 → 空串；小数截成整数）
_json_num() {
    local v
    v=$(grep -oE "\"$2\": *[0-9.]+" "$1" 2>/dev/null | head -1 | grep -oE '[0-9.]+$' || true)
    echo "${v%.*}"
    return 0
}

# _cell_k <结果目录> <key>  →  打印「K 真实峰值|- 可合包(0/1) 说明」；K=0 表示还没探路
#   读本格子的包，以及本格子参与过的跨格子合包（<group>__mix-a+b+…__pack-*，D-060）
_cell_k() {
    local dir="$1" key="$2" f k total peak memmax noom
    local peak_max="" total_max="" est="" oom_cap="" proven=0 n_files=0 warn=0 w
    local group="${key%%__*}" cell="${key#*__}" mid
    for f in "$dir/${key}__pack-"*.gpu.json "$dir/${group}__mix-"*"__pack-"*.gpu.json; do
        [ -f "$f" ] || continue
        case "${f##*/}" in
            "${group}__mix-"*)               # 合包：文件名里的格子列表要含本格子
                mid="${f##*/}"; mid="${mid#"${group}__mix-"}"; mid="${mid%%__pack-*}"
                [[ "+$mid+" == *"+$cell+"* ]] || continue ;;
        esac
        n_files=$((n_files + 1))
        k=$(_json_num "$f" k); total=$(_json_num "$f" mem_total_mib)
        peak=$(_json_num "$f" run_peak_max_mib); memmax=$(_json_num "$f" mem_max_mib)
        noom=$(_json_num "$f" n_oom); w=$(_json_num "$f" n_mem_warnings)
        [ -n "$w" ] && warn=$((warn + w))
        [ -n "$total" ] && { [ -z "$total_max" ] || [ "$total" -gt "$total_max" ]; } && total_max=$total
        if [ -n "$peak" ]; then
            { [ -z "$peak_max" ] || [ "$peak" -gt "$peak_max" ]; } && peak_max=$peak
        elif [ -n "$memmax" ] && [ -n "$k" ] && [ "$k" -gt 0 ]; then
            local e=$((memmax / k))
            { [ -z "$est" ] || [ "$e" -gt "$est" ]; } && est=$e
        fi
        if [ -n "$noom" ] && [ "$noom" -gt 0 ]; then
            local cap=$((k - 1))
            { [ -z "$oom_cap" ] || [ "$cap" -lt "$oom_cap" ]; } && oom_cap=$cap
        elif [ -n "$k" ] && [ "$k" -gt "$proven" ]; then
            proven=$k
        fi
    done
    if [ "$n_files" -eq 0 ]; then echo "0 - 0 还没有显存数据"; return 0; fi

    local kc=$PACK why="" per="${peak_max:-$est}"
    if [ -n "$total_max" ] && [ -n "$per" ]; then
        local kmem=$(( total_max * PACK_MEM_PCT / 100 / (per + PACK_CTX_MIB) ))
        [ "$kmem" -lt "$kc" ] && kc=$kmem
        why="每 run 峰值=${per}MiB$([ -z "$peak_max" ] && echo '(整卡读数/k)') 容量=${total_max}MiB×${PACK_MEM_PCT}%"
    else
        [ "$proven" -gt 0 ] && [ "$proven" -lt "$kc" ] && kc=$proven
        why="没有容量/峰值读数 → 不超过跑通过的 k=$proven"
    fi
    if [ -n "$oom_cap" ]; then
        [ "$oom_cap" -lt "$kc" ] && kc=$oom_cap
        why="$why；出过 OOM → ≤ $oom_cap"
    fi
    [ "$kc" -lt 1 ] && kc=1
    [ "$warn" -gt 0 ] && why="$why；显存告警 ${warn} 次（未降档，见 D-052）"
    # 余数能不能跨格子合包（D-060）：K 来自真实峰值、没有 OOM、没有显存告警
    local mixable=0
    [ -n "$peak_max" ] && [ -z "$oom_cap" ] && [ "$warn" -eq 0 ] && mixable=1
    echo "$kc ${peak_max:--} $mixable $why"
    return 0
}

# _submit_pack <key> <下标…>   交一个包（或 dry-run 打印）
_submit_pack() {
    local key="$1"; shift
    local n=$# args=() rids=() i
    for i in "$@"; do
        args+=("${_PK_CFG[$i]}" "${_PK_RID[$i]}" "${_PK_MET[$i]}"); rids+=("${_PK_RID[$i]}")
    done
    local tag="${key}__pack-k${n}-s${_PK_SEED[$1]}"
    if [ "$SUBMIT_OK" -eq 1 ]; then
        (cd "$ROOT" && sbatch -c $((4 * n)) --job-name="exp3v2-pack$n" "$PACK_JOB" "$tag" "${args[@]}")
        queued=$((queued + n))
    else
        echo "  would sbatch -c $((4 * n)) pack  $tag  ${rids[*]}"
    fi
    return 0
}

# pack_flush：把分好桶的待交 run 按上面的规则交出去
pack_flush() {
    local key idx kc peak mixable why dir
    [ "${#_PK_KEYS[@]}" -eq 0 ] && return 0
    _MIX_KEYS=(); _MIX_REM=(); _MIX_K=(); _MIX_PEAK=()
    for key in "${_PK_KEYS[@]}"; do
        read -ra idx <<< "${_PK_BUCKET[$key]}"
        dir=$(dirname "${_PK_MABS[${idx[0]}]}")
        read -r kc peak mixable why <<< "$(_cell_k "$dir" "$key")"
        if [ "$kc" -eq 0 ]; then
            local p=$PROBE_K
            [ "$PACK" -lt "$p" ] && p=$PACK
            [ "${#idx[@]}" -lt "$p" ] && p=${#idx[@]}
            echo "  [pack] $key：探路 k=$p（$why）；其余 $(( ${#idx[@]} - p )) 个 held，显存数据回来后再跑一次"
            _submit_pack "$key" "${idx[@]:0:$p}"
            held=$((held + ${#idx[@]} - p))
            continue
        fi
        echo "  [pack] $key：K=$kc（$why）"
        local s nfull=$(( ${#idx[@]} / kc * kc ))
        for ((s = 0; s < nfull; s += kc)); do
            _submit_pack "$key" "${idx[@]:$s:$kc}"
        done
        [ "$nfull" -eq "${#idx[@]}" ] && continue
        if [ "$PACK_MIX" -eq 1 ] && [ "$mixable" -eq 1 ]; then      # 余数先挂起，下面按组试着合包
            _MIX_KEYS+=("$key"); _MIX_REM[$key]="${idx[*]:$nfull}"
            _MIX_K[$key]=$kc; _MIX_PEAK[$key]=$peak
        else
            _submit_pack "$key" "${idx[@]:$nfull}"
        fi
    done
    _mix_flush
    return 0
}

# _mix_flush：各格子挂起的余数，按组跨格子合包（D-060；规则见页首）。不合的照旧各交各的。
_mix_flush() {
    [ "${#_MIX_KEYS[@]}" -eq 0 ] && return 0
    local groups=() g key r
    for key in "${_MIX_KEYS[@]}"; do
        g=${key%%__*}
        [[ " ${groups[*]} " == *" $g "* ]] || groups+=("$g")
    done
    for g in "${groups[@]}"; do
        local ks=() kmin="" pmin="" pmax="" total=0 reason=""
        for key in "${_MIX_KEYS[@]}"; do
            [ "${key%%__*}" = "$g" ] || continue
            ks+=("$key")
            read -ra r <<< "${_MIX_REM[$key]}"
            total=$((total + ${#r[@]}))
            { [ -z "$kmin" ] || [ "${_MIX_K[$key]}" -lt "$kmin" ]; } && kmin=${_MIX_K[$key]}
            { [ -z "$pmin" ] || [ "${_MIX_PEAK[$key]}" -lt "$pmin" ]; } && pmin=${_MIX_PEAK[$key]}
            { [ -z "$pmax" ] || [ "${_MIX_PEAK[$key]}" -gt "$pmax" ]; } && pmax=${_MIX_PEAK[$key]}
        done
        local merged=$(( (total + kmin - 1) / kmin ))
        if [ "${#ks[@]}" -lt 2 ]; then
            reason="only-one"
        elif [ $(( (pmax - pmin) * 100 )) -gt $(( PACK_MIX_TOL_PCT * pmax )) ]; then
            reason="真实峰值 ${pmin}–${pmax}MiB 相差超过 ${PACK_MIX_TOL_PCT}%"
        elif [ "$merged" -ge "${#ks[@]}" ]; then
            reason="合了也要 ${merged} 个作业，不省"
        fi
        if [ -n "$reason" ]; then
            [ "$reason" != only-one ] && echo "  [pack] $g：余数不跨格子合包（$reason）"
            for key in "${ks[@]}"; do
                read -ra r <<< "${_MIX_REM[$key]}"
                _submit_pack "$key" "${r[@]}"
            done
            continue
        fi
        echo "  [pack] $g：${#ks[@]} 个格子的余数跨格子合包 K=$kmin（真实峰值 ${pmin}–${pmax}MiB，相差 ≤ ${PACK_MIX_TOL_PCT}%；${#ks[@]} → ${merged} 个作业，D-060）"
        local pool=()
        for key in "${ks[@]}"; do
            read -ra r <<< "${_MIX_REM[$key]}"
            pool+=("${r[@]}")
        done
        local s
        for ((s = 0; s < ${#pool[@]}; s += kmin)); do
            local chunk=("${pool[@]:$s:$kmin}") cells=() i
            for i in "${chunk[@]}"; do
                [[ " ${cells[*]} " == *" ${_PK_CELL[$i]} "* ]] || cells+=("${_PK_CELL[$i]}")
            done
            if [ "${#cells[@]}" -eq 1 ]; then
                _submit_pack "${g}__${cells[0]}" "${chunk[@]}"
            else
                _submit_pack "${g}__mix-$(IFS=+; echo "${cells[*]}")" "${chunk[@]}"
            fi
        done
    done
    return 0
}
