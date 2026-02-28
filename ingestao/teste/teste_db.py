from ingestao.db.banco_metadados import MetadataDB


metadata = MetadataDB()


# ======================================
# TESTE 1 - Busca por autor
# ======================================

def teste_db_autor(autor):
    print("\n=== TESTE: BUSCAR POR AUTOR ===")
    base = metadata.buscar_autor(autor)

    print(f"Total encontrados: {len(base)}")

    for doc in base:
        print(f"- {doc['titulo']} ({doc['ano']})")
        print(f"  ID: {doc['id']}")
        print(f"  Status: {doc['status_ingestao']}")
        print()


# ======================================
# TESTE 2 - Busca pendentes por autor
# ======================================

def teste_pendentes_por_autor(autor):
    print("\n=== TESTE: PENDENTES POR AUTOR ===")
    base = metadata.buscar_pendentes_por_autor(autor)

    print(f"Pendentes encontrados: {len(base)}")

    for doc in base:
        print(f"- {doc['titulo']} | Status: {doc['status_ingestao']}")


# ======================================
# TESTE 3 - Buscar documento por ID
# ======================================

def teste_buscar_por_id(doc_id):
    print("\n=== TESTE: BUSCAR POR ID ===")
    doc = metadata.buscar_documento(doc_id)

    if not doc:
        print("Documento não encontrado.")
        return

    print(f"Título: {doc['titulo']}")
    print(f"Autor(es): {doc['autores']}")
    print(f"Ano: {doc['ano']}")
    print(f"Status: {doc['status_ingestao']}")


# ======================================
# TESTE 4 - Atualizar status
# ======================================

def teste_atualizar_status(doc_id, novo_status):
    print("\n=== TESTE: ATUALIZAR STATUS ===")
    metadata.atualizar_status(doc_id, novo_status)

    doc = metadata.buscar_documento(doc_id)
    print(f"Novo status: {doc['status_ingestao']}")


# ======================================
# TESTE 5 - Buscar erros
# ======================================

def teste_buscar_erros():
    print("\n=== TESTE: DOCUMENTOS COM ERRO ===")
    erros = metadata.buscar_erros()

    print(f"Total com erro: {len(erros)}")

    for doc in erros:
        print(f"- {doc['titulo']} | ID: {doc['id']}")


# ======================================
# TESTE 6 - Estatísticas gerais
# ======================================

def teste_estatisticas():
    print("\n=== TESTE: ESTATÍSTICAS GERAIS ===")

    todos = metadata.buscar_autor("")  # busca tudo
    total = len(todos)

    processados = len([d for d in todos if d["status_ingestao"] == "processado"])
    pendentes = len([d for d in todos if d["status_ingestao"] == "pendente"])
    erros = len([d for d in todos if d["status_ingestao"] == "erro"])

    print(f"Total documentos: {total}")
    print(f"Processados: {processados}")
    print(f"Pendentes: {pendentes}")
    print(f"Erros: {erros}")


# ======================================
# EXECUÇÃO
# ======================================

if __name__ == "__main__":

    autor = "Danilo Santa Cruz Coelho"

    teste_db_autor(autor)
    teste_pendentes_por_autor(autor)

    # Se existir algum documento do autor, usa o primeiro para testes
    docs = metadata.buscar_autor(autor)
    if docs:
        doc_id = docs[0]["id"]

        teste_buscar_por_id(doc_id)
        teste_atualizar_status(doc_id, "pendente")  # apenas para teste

    teste_buscar_erros()
    teste_estatisticas()

    print("\n🔥 Testes concluídos.")