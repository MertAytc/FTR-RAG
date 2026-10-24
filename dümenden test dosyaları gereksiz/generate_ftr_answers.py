import argparse, json, time, urllib.request, urllib.error
from pathlib import Path

OLLAMA_URL = 'http://localhost:11434/api/chat'

SYSTEM_PROMPT = '''Sen FTR-RAG sisteminin değerlendirme aşamasında kullanılan, kanıta bağlı bir fizyoterapi bilgi asistanısın.
Yalnızca verilen bağlam parçalarına dayanarak soruyu Türkçe yanıtla.
- Bağlamda yeterli kanıt yoksa tahmin yürütme; açıkça "Verilen kaynak parçalarında bu soruyu yanıtlamak için yeterli bilgi bulunamadı." de.
- Kaynakta belirtilmeyen doz, süre, klinik eşik veya öneri uydurma.
- Koşulları, istisnaları ve kanıtın kapsamını koru.
- Kısa ve doğrudan cevap ver; gerekiyorsa 1-3 madde kullan.
- Tanı veya kişiye özel tedavi talimatı verme; kılavuz bilgisini özetle.'''


def context_to_text(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        # Common field names used by retrieval datasets
        for key in ('text', 'content', 'page_content', 'chunk_text', 'document'):
            if isinstance(item.get(key), str) and item[key].strip():
                return item[key]
        return json.dumps(item, ensure_ascii=False)
    return str(item)


def ollama_chat(model, question, contexts, timeout):
    # Use a bounded context window to avoid huge prompts; retain full contexts in output dataset.
    selected = contexts[:8]
    context_block = '\n\n'.join(f'[Bağlam {i+1}]\n{context_to_text(c)}' for i, c in enumerate(selected))
    user_prompt = f'''Soru: {question}

Getirilen kaynak bağlamları (yalnızca bunları kullan):
{context_block}

Kaynaklara dayalı cevabın:'''
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt}
        ],
        'stream': False,
        'options': {'temperature': 0, 'num_ctx': 8192}
    }
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode('utf-8'))
    return result['message']['content'].strip()


def main():
    parser = argparse.ArgumentParser(description='Generate answers for FTR-RAG retrieval benchmark records using local Ollama.')
    parser.add_argument('--input', default='retrieval_ragas_v2_final_run2_ragas_retrieval_dataset.jsonl')
    parser.add_argument('--output', default='retrieval_ragas_v2_final_run2_answers_qwen3_8b.jsonl')
    parser.add_argument('--model', default='qwen3:8b')
    parser.add_argument('--timeout', type=int, default=240)
    parser.add_argument('--limit', type=int, default=0, help='Optional small test limit; 0 means all records.')
    parser.add_argument('--retry', type=int, default=2)
    args = parser.parse_args()

    in_path, out_path = Path(args.input), Path(args.output)
    records = []
    with in_path.open('r', encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f'JSONL parse error at line {line_no}: {exc}')
    if args.limit > 0:
        records = records[:args.limit]
    if not records:
        raise SystemExit('No records found in input.')

    # Write each completed record immediately so progress is not lost on interruption.
    done = 0
    with out_path.open('w', encoding='utf-8', newline='\n') as out:
        for idx, record in enumerate(records, 1):
            question = record.get('question') or ''
            contexts = record.get('contexts') or []
            if not question.strip():
                record['answer'] = ''
                record['answer_status'] = 'error_missing_question'
            else:
                last_error = None
                for attempt in range(args.retry + 1):
                    try:
                        answer = ollama_chat(args.model, question, contexts, args.timeout)
                        record['answer'] = answer
                        record['answer_status'] = 'generated'
                        record['answer_model'] = args.model
                        record['answer_contexts_used'] = min(8, len(contexts))
                        last_error = None
                        break
                    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                        last_error = str(exc)
                        if attempt < args.retry:
                            time.sleep(2 * (attempt + 1))
                if last_error is not None:
                    record['answer'] = ''
                    record['answer_status'] = 'generation_error'
                    record['answer_note'] = last_error
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            out.flush()
            done += 1
            status = record.get('answer_status', '')
            print(f'[{idx}/{len(records)}] {record.get("question_id", "?")} | {record.get("method", "?")} | {status}', flush=True)

    print(f'\nTamamlandı: {done} kayıt')
    print(f'Çıktı: {out_path.resolve()}')
    print('Not: Bu işlem model cevaplarını üretir; gold/reference cevaplarını üretmez.')

if __name__ == '__main__':
    main()
