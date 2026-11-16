import time, requests

URL = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
s = requests.Session()
s.trust_env = False   # 不走学术加速代理，直接连本机

def make_doc(word):
    # 大约 2000 个 token 的"文章"，word 不同就是不同的文章
    return " ".join(f"Sentence {i}: the quick brown {word} jumps over the lazy dog." for i in range(150))

def ttft_ms(prompt):
    t0 = time.perf_counter()
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 16, "temperature": 0, "stream": True}
    with s.post(URL + "/v1/completions", json=body, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith(b"data:") and b"[DONE]" not in line:
                return (time.perf_counter() - t0) * 1000   # 收到第一个 token 的时间

def prefix_metrics():
    lines = s.get(URL + "/metrics").text.splitlines()
    return [l for l in lines if l.startswith("vllm:prefix_cache") and not l.startswith("#")]

ttft_ms("warm up")  # 先热身一次，排除首次请求的额外开销

doc1, doc2 = make_doc("fox"), make_doc("cat")
q1 = "\nQuestion: what animal is mentioned? Answer:"
q2 = "\nQuestion: how many sentences are there? Answer:"

print("测试前:", *prefix_metrics(), sep="\n  ")
a = ttft_ms(doc1 + q1); print(f"A 文章1+问题1 (冷):   {a:6.1f} ms")
b = ttft_ms(doc1 + q2); print(f"B 文章1+问题2 (复用): {b:6.1f} ms")
c = ttft_ms(doc2 + q1); print(f"C 文章2+问题1 (冷):   {c:6.1f} ms")
print("测试后:", *prefix_metrics(), sep="\n  ")
print(f"\nB 比 A 快 {a / b:.1f} 倍；C/A = {c / a:.2f}（应接近 1）")
