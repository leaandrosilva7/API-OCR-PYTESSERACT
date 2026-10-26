# =====================================================================
# API de OCR para o boletim do ENEM
# Recebe um PDF, devolve o texto dele. Se o PDF for só uma imagem
# (print), usa o Tesseract para ler o texto da imagem.
# =====================================================================

import os
import secrets
import threading

import numpy as np
import pymupdf as fitz  # PyMuPDF: abre o PDF e transforma páginas em imagem
import pytesseract
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, ImageOps


# =====================================================================
# 1. CONFIGURAÇÕES
# Valores que você pode ajustar sem mexer no resto do código
# =====================================================================

# Só descomente no Windows se o tesseract não for encontrado no PATH
# (no servidor Linux/Docker, deixe comentado)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TOKEN_DA_API = os.environ.get("OCR_TOKEN", "")        # senha da API
PERMITIR_SEM_TOKEN = os.environ.get("OCR_SEM_TOKEN") == "1"  # só para teste local

DPI = 300                            # qualidade da imagem gerada a partir do PDF
IDIOMA = "por"                       # idioma do Tesseract
MAX_PAGINAS = 5                      # só lê as primeiras páginas do PDF
MAX_BYTES = 10 * 1024 * 1024         # tamanho máximo do arquivo (10 MB)
MIN_CHARS_TEXTO_NATIVO = 50          # abaixo disso, considera que o PDF não tem texto útil
MAX_PIXELS_POR_PAGINA = 40_000_000   # protege a memória (uma A4 a 300 DPI tem ~8,7 milhões)
TEMPO_MAX_OCR = 60                   # segundos máximos por leitura do Tesseract
VAGAS_OCR = threading.BoundedSemaphore(2)  # no máximo 2 OCRs ao mesmo tempo

# Sem token e sem liberação explícita, a API não sobe
if not TOKEN_DA_API and not PERMITIR_SEM_TOKEN:
    raise RuntimeError(
        "Defina a variável OCR_TOKEN antes de subir a API "
        "(ou OCR_SEM_TOKEN=1 para testes locais sem senha)."
    )


# =====================================================================
# 2. CRIAÇÃO DA API
# Se houver token, a página /docs é desligada por segurança
# =====================================================================

app = FastAPI(
    title="OCR Boletim ENEM",
    docs_url=None if TOKEN_DA_API else "/docs",
    redoc_url=None,
    openapi_url=None if TOKEN_DA_API else "/openapi.json",
)


# =====================================================================
# 3. SEGURANÇA: confere o token enviado no cabeçalho "X-Api-Key"
# Se não bater com o OCR_TOKEN, devolve erro 401
# =====================================================================

def checa_token(x_api_key: str = Header(default="")):
    if not TOKEN_DA_API:
        return  # modo teste local sem senha
    if not secrets.compare_digest(x_api_key.encode(), TOKEN_DA_API.encode()):
        raise HTTPException(status_code=401, detail="Não autorizado")


# =====================================================================
# 4. PDF -> IMAGEM
# Transforma uma página do PDF em imagem em tons de cinza
# =====================================================================

def pagina_para_imagem(pagina) -> Image.Image:
    zoom = DPI / 72  # o PDF trabalha em 72 DPI; isso converte para 300

    # Recusa páginas enormes antes de gerar a imagem (evita estourar a memória)
    if pagina.rect.width * zoom * pagina.rect.height * zoom > MAX_PIXELS_POR_PAGINA:
        raise HTTPException(status_code=422, detail="Página grande demais")

    pixmap = pagina.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        colorspace=fitz.csGRAY,
        alpha=False,
    )
    return Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)


# =====================================================================
# 5. LIMIAR DE OTSU
# Descobre automaticamente o "ponto de corte" entre o que é fundo e o
# que é letra (usado para deixar a imagem só preto e branco).
# É um cálculo matemático padrão; não precisa mexer nele.
# =====================================================================

def calcula_limiar_otsu(pixels: np.ndarray) -> int:
    contagem = np.bincount(pixels.ravel(), minlength=256).astype(float)
    total = pixels.size
    peso_fundo = np.cumsum(contagem) / total
    media_acumulada = np.cumsum(contagem * np.arange(256)) / total

    with np.errstate(divide="ignore", invalid="ignore"):
        separacao = (media_acumulada[-1] * peso_fundo - media_acumulada) ** 2 / (
            peso_fundo * (1 - peso_fundo)
        )
    return int(np.nanargmax(separacao))


# =====================================================================
# 6. PREPARAÇÃO DA IMAGEM
# Gera 3 versões da imagem para o Tesseract tentar ler:
#   1) a imagem original (já ajustada)
#   2) com contraste aumentado
#   3) só preto e branco (método de Otsu)
# =====================================================================

def gera_versoes_da_imagem(imagem: Image.Image) -> list:
    # Tema escuro (fundo escuro, letra clara): inverte as cores
    if np.asarray(imagem).mean() < 127:
        imagem = ImageOps.invert(imagem)

    # Print pequeno: amplia 2x para o Tesseract enxergar melhor
    if imagem.width < 2000:
        imagem = imagem.resize((imagem.width * 2, imagem.height * 2), Image.LANCZOS)

    versao_original = imagem
    versao_contraste = ImageOps.autocontrast(imagem, cutoff=1)

    pixels = np.asarray(imagem)
    limiar = calcula_limiar_otsu(pixels)
    versao_preto_branco = Image.fromarray(((pixels > limiar) * 255).astype(np.uint8))

    return [versao_original, versao_contraste, versao_preto_branco]


# =====================================================================
# 7. OCR (leitura da imagem)
# Roda o Tesseract nas 3 versões e fica com a que ele leu com mais
# confiança. Devolve (texto, confiança média de 0 a 100).
# Cada leitura tem tempo máximo (TEMPO_MAX_OCR).
# =====================================================================

def le_imagem_com_ocr(imagem: Image.Image):
    config = "--oem 3 --psm 6"
    melhor_texto = ""
    melhor_confianca = -1.0

    for versao in gera_versoes_da_imagem(imagem):
        # Pede ao Tesseract a confiança de cada palavra lida
        dados = pytesseract.image_to_data(
            versao,
            lang=IDIOMA,
            config=config,
            output_type=pytesseract.Output.DICT,
            timeout=TEMPO_MAX_OCR,
        )
        confiancas = [float(c) for c in dados["conf"] if float(c) >= 0]
        confianca_media = sum(confiancas) / len(confiancas) if confiancas else 0.0

        # Se essa versão foi melhor que as anteriores, guarda o texto dela
        if confianca_media > melhor_confianca:
            melhor_confianca = confianca_media
            melhor_texto = pytesseract.image_to_string(
                versao, lang=IDIOMA, config=config, timeout=TEMPO_MAX_OCR
            )

    return melhor_texto, max(melhor_confianca, 0.0)


# =====================================================================
# 8. ROTA PRINCIPAL: POST /ocr
# É a que o n8n chama. Recebe o PDF e devolve o texto.
# (def normal, não async: o OCR é pesado e assim não trava a API toda)
# =====================================================================

@app.post("/ocr", dependencies=[Depends(checa_token)])
def ocr(file: UploadFile = File(...)):

    # --- 8.1 Lê o arquivo enviado e confere o tamanho ---
    dados = file.file.read(MAX_BYTES + 1)
    if len(dados) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Arquivo muito grande")

    # --- 8.2 Abre o PDF (erro 422 se estiver corrompido ou com senha) ---
    try:
        pdf = fitz.open(stream=dados, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=422, detail="PDF inválido ou corrompido")

    if pdf.needs_pass:
        raise HTTPException(status_code=422, detail="PDF protegido por senha")

    # --- 8.3 Lê página por página ---
    textos = []
    confiancas = []
    metodo = "texto_nativo"

    for pagina in list(pdf)[:MAX_PAGINAS]:
        # Primeiro tenta o texto que já existe dentro do PDF
        texto_nativo = pagina.get_text().strip()
        if len(texto_nativo) >= MIN_CHARS_TEXTO_NATIVO:
            textos.append(texto_nativo)
            continue

        # Se não tem texto (PDF de print), usa o OCR
        metodo = "ocr"
        imagem = pagina_para_imagem(pagina)
        try:
            with VAGAS_OCR:  # espera vaga se já houver 2 OCRs rodando
                texto_ocr, confianca = le_imagem_com_ocr(imagem)
        except RuntimeError as erro:
            if "timeout" in str(erro).lower():
                raise HTTPException(status_code=504, detail="OCR demorou demais")
            raise
        textos.append(texto_ocr)
        confiancas.append(confianca)

    # --- 8.4 Monta a resposta (a chave "text" é a que o n8n lê) ---
    confianca_media = round(sum(confiancas) / len(confiancas), 1) if confiancas else None

    return {
        "text": "\n".join(textos).strip(),
        "metodo": metodo,
        "confianca": confianca_media,
        "paginas": len(textos),
    }


# =====================================================================
# 9. ROTA DE SAÚDE: GET /health
# Só para conferir se a API e o Tesseract estão funcionando
# =====================================================================

@app.get("/health")
def health():
    return {"ok": True, "tesseract": str(pytesseract.get_tesseract_version())}