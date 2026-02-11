from ingestao import create_ingestion
from scraper import Scraper

import time
import os

TOTAL_PAGES = 860

if __name__ == "__main__":

    # scraper = Scraper()
    # for i in range(1, 860):
    #     scraper.processar_pagina(i)
    #
    # time.sleep(5)

    if os.name == "nt":
        os.system('cls')

    while True:
        try:
            sucesso = create_ingestion.processar_documento()
            if not sucesso:
                break
        except Exception as e:
            print(f"Erro ao processar documento. Continuando com o próximo... {e}")
            continue
    print("Pipeline concluído.")



