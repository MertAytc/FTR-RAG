from src.retrieval.jev_reranker import JevReranker


query = "WHO'nun kronik primer bel ağrısı için rehabilitasyon paketinin hedef popülasyonu nasıl tanımlanmaktadır?"

document = """
Chronic primary low back pain rehabilitation package:
The target population includes adults with chronic primary low back pain,
with or without associated leg pain.
"""


reranker = JevReranker()

score = reranker._score_candidate(
    query=query,
    document=document,
)

print()
print("=" * 60)
print("JEV TEST")
print("=" * 60)
print(f"Query : {query}")
print(f"Score : {score}")
print("=" * 60)