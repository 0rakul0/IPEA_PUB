from pathlib import Path
import sqlite3
from typing import Any, Dict, Optional

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "banco1.db"

class MetadataDB:
    """
    Wrapper simples em torno de um banco SQLite para metadados de documentos.

    Os métodos espelham as funções de nível de módulo anteriores:
    - conectar -> abre conexão
    - criar_tabela -> cria a tabela principal
    - inserir_documento, buscar_documento, atualizar_documento
    - buscar_pendente, atualizar_status
    """

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.criar_tabela()

    def conectar(self) -> sqlite3.Connection:
        """Abre uma conexão com row_factory configurado para sqlite3.Row."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def criar_tabela(self) -> None:
        """Cria a tabela 'documentos' caso não exista."""
        with self.conectar() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documentos (
                    id TEXT PRIMARY KEY,
                    titulo TEXT,
                    autores TEXT,
                    ano INTEGER,
                    tipo_conteudo TEXT,
                    resumo TEXT,
                    palavras_chave TEXT,
                    link_pdf TEXT,
                    link_download TEXT,
                    status_ingestao TEXT,
                    data_ingestao TEXT
                );
            """)
            conn.commit()

    def remover_duplicatas(self) -> int:
        """
        Remove documentos duplicados com base em (titulo, ano, resumo).

        Mantém o registro com menor rowid (mais antigo inserido).
        Retorna o número de registros removidos.
        """
        with self.conectar() as conn:
            cursor = conn.cursor()

            # 1️⃣ Identificar grupos duplicados
            cursor.execute("""
                SELECT 
                    MIN(rowid) as keep_rowid,
                    GROUP_CONCAT(rowid) as all_rowids,
                    COUNT(*) as total
                FROM documentos
                WHERE titulo IS NOT NULL
                  AND resumo IS NOT NULL
                  AND ano IS NOT NULL
                GROUP BY 
                    LOWER(TRIM(titulo)),
                    ano,
                    LOWER(TRIM(resumo))
                HAVING COUNT(*) > 1
            """)

            duplicados = cursor.fetchall()

            total_removidos = 0

            for row in duplicados:
                keep_rowid = row["keep_rowid"]
                all_rowids = row["all_rowids"].split(",")

                # remover todos exceto o keep_rowid
                rowids_para_remover = [r for r in all_rowids if int(r) != keep_rowid]

                if rowids_para_remover:
                    cursor.execute(f"""
                        DELETE FROM documentos
                        WHERE rowid IN ({','.join(['?'] * len(rowids_para_remover))})
                    """, rowids_para_remover)

                    total_removidos += len(rowids_para_remover)

            conn.commit()

            return total_removidos


    def inserir_documento(self, document: Dict[str, Any]) -> None:
        """Insere ou substitui um documento por id."""
        with self.conectar() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documentos (
                    id, titulo, autores, ano, tipo_conteudo,
                    resumo, palavras_chave, link_pdf, link_download,
                    status_ingestao, data_ingestao
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (

                document.get("id"),
                document.get("titulo"),
                document.get("autores"),
                document.get("ano"),
                document.get("tipo_conteudo"),
                document.get("resumo"),
                document.get("palavras_chave"),
                document.get("link_pdf"),
                document.get("link_download"),
                document.get("status_ingestao"),
                document.get("data_ingestao"),
            ))
            conn.commit()

    def buscar_documento(self, id: str) -> Optional[Dict[str, Any]]:
        """Retorna o documento como dict ou None se não existir."""
        with self.conectar() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documentos WHERE id = ?", (id,))
            row = cursor.fetchone()
        return dict(row) if row else None

    def atualizar_documento(self, document: Dict[str, Any]) -> None:
        """Atualiza campos do documento identificado por id."""
        with self.conectar() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE documentos
                SET titulo = ?, autores = ?, ano = ?, tipo_conteudo = ?,
                    resumo = ?, palavras_chave = ?, link_pdf = ?, link_download = ?,
                    status_ingestao = ?, data_ingestao = ?
                WHERE id = ?
            """, (
                document.get("titulo"),
                document.get("autores"),
                document.get("ano"),
                document.get("tipo_conteudo"),
                document.get("resumo"),
                document.get("palavras_chave"),
                document.get("link_pdf"),
                document.get("link_download"),
                document.get("status_ingestao"),
                document.get("data_ingestao"),
                document.get("id"),
            ))
            conn.commit()

    def buscar_pendente(self, randomize: bool = False) -> Optional[Dict[str, Any]]:
        """
        Busca e reserva um documento pendente
        """
        with self.conectar() as conn:
            cursor = conn.cursor()

            if randomize:
                query = """
                    SELECT * 
                    FROM documentos 
                    WHERE status_ingestao = 'pendente'
                    ORDER BY RANDOM()
                    LIMIT 1
                """
            else:
                query = """
                    SELECT * 
                    FROM documentos 
                    WHERE status_ingestao = 'pendente'
                    ORDER BY id ASC
                    LIMIT 1
                """

            cursor.execute(query)
            row = cursor.fetchone()
            return dict(row) if row else None

    def atualizar_status(self, id: str, status: str) -> None:
        """Atualiza apenas o status_ingestao do documento."""
        with self.conectar() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE documentos
                SET status_ingestao = ?
                WHERE id = ?
            """, (status, id))
            conn.commit()

    def atualizar_link_donwload(self, id: str, link_download: str) -> None:
        """Atualiza apenas o status_ingestao do documento."""
        with self.conectar() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE documentos
                SET link_download = ?
                WHERE id = ?
            """, (link_download, id))
            conn.commit()
