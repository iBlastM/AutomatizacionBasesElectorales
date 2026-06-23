from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .perfiles import PerfilFormato, TOP_COLUMNS, normalizar_clave
from .configuracion import ConfiguracionAnual, config_desde_perfil_simple


@dataclass
class ResultadoFormateoSimple:
    df_base: pd.DataFrame
    config: ConfiguracionAnual
    advertencias: list[str]


class FormateadorSimple:
    def __init__(self, perfil: PerfilFormato):
        self.perfil = perfil

    def formatear(self, tabla: pd.DataFrame) -> ResultadoFormateoSimple:
        if not self.perfil.es_simple:
            raise ValueError("FormateadorSimple solo acepta perfiles de ayuntamientos o gubernatura.")

        advertencias: list[str] = []
        df = tabla.copy()
        df.columns = [str(col).strip() for col in df.columns]

        trabajo = pd.DataFrame({"SECCION": self._serie_numerica(df, ["SECCION"])})
        trabajo["__MUNICIPIO"] = self._serie_texto(df, self.perfil.aliases_columnas["municipio"])
        trabajo["__LISTA"] = self._serie_numerica_logica(df, "lista", advertencias)
        trabajo["__VOTOS"] = self._serie_numerica_logica(df, "votos", advertencias)
        trabajo["__NULOS"] = self._serie_numerica_logica(df, "nulos", advertencias)

        for idx, partido in enumerate(self.perfil.partidos_salida):
            trabajo[f"__P_{idx}"] = self._serie_numerica_logica(df, partido, advertencias)

        for idx, (etiqueta, componentes) in enumerate(self.perfil.ranking_grupos):
            trabajo[f"__R_{idx}"] = self._sumar_componentes(df, componentes, advertencias)

        agregaciones = {
            "__MUNICIPIO": self._primero_no_vacio,
            "__LISTA": "sum",
            "__VOTOS": "sum",
            "__NULOS": "sum",
        }
        for idx in range(len(self.perfil.partidos_salida)):
            agregaciones[f"__P_{idx}"] = "sum"
        for idx in range(len(self.perfil.ranking_grupos)):
            agregaciones[f"__R_{idx}"] = "sum"

        agrupado = trabajo.groupby("SECCION", as_index=False).agg(agregaciones)
        agrupado = agrupado.sort_values("SECCION").reset_index(drop=True)

        config = config_desde_perfil_simple(self.perfil)
        usar_ranking = self._usar_ranking_referencia(agrupado)
        filas = [self._fila_salida(row, usar_ranking) for _, row in agrupado.iterrows()]
        resultado = pd.DataFrame(filas, columns=self.perfil.encabezados_visibles)

        return ResultadoFormateoSimple(df_base=resultado, config=config, advertencias=advertencias)

    def _fila_salida(self, row: pd.Series, usar_ranking_referencia: bool) -> dict[str, object]:
        salida: dict[str, object] = {col: "" for col in self.perfil.encabezados_visibles}
        municipio_col = self._columna_salida(["MUNICIPIO"])
        seccion_col = self._columna_salida(["SECCION", "Seccion"])
        lista_col = self._columna_salida(["LISTA_NOMINAL", "Lista Nominal"])
        votos_col = self._columna_salida(["VOTOS_EMITIDOS", "Votos Emitidos"])
        participacion_col = self._columna_salida(["PARTICIPACION_PCN", "PARTICIPACION", "Participacion (%)"])
        nulos_col = self._columna_salida(["NULOS", "Nulos"])

        lista = self._normalizar_numero(row["__LISTA"])
        votos = self._normalizar_numero(row["__VOTOS"])
        nulos = self._normalizar_numero(row["__NULOS"])

        salida[municipio_col] = row["__MUNICIPIO"]
        salida[seccion_col] = int(row["SECCION"])
        salida[lista_col] = lista
        salida[votos_col] = votos
        salida[participacion_col] = votos / lista if lista else 0
        salida[nulos_col] = nulos

        for idx, partido in enumerate(self.perfil.partidos_salida):
            salida[partido] = self._normalizar_numero(row[f"__P_{idx}"])

        seccion = int(row["SECCION"])
        if usar_ranking_referencia and seccion in self.perfil.ranking_referencia:
            ranking = [
                (etiqueta, votos, idx)
                for idx, (etiqueta, votos) in enumerate(self.perfil.ranking_referencia[seccion])
            ]
        else:
            ranking = []
            for idx, (etiqueta, _) in enumerate(self.perfil.ranking_grupos):
                valor = self._normalizar_numero(row[f"__R_{idx}"])
                if valor > 0:
                    ranking.append((etiqueta, valor, idx))
            ranking.sort(key=lambda item: (-item[1], item[2]))
            ranking = self._seleccionar_top_sin_duplicar_componentes(ranking)

        top_cols = TOP_COLUMNS[self.perfil.id]
        for orden in range(3):
            etiqueta_col = top_cols[orden * 2]
            votos_top_col = top_cols[orden * 2 + 1]
            if orden < len(ranking):
                salida[etiqueta_col] = ranking[orden][0]
                salida[votos_top_col] = ranking[orden][1]
            else:
                salida[etiqueta_col] = ""
                salida[votos_top_col] = 0
        return salida

    def _usar_ranking_referencia(self, agrupado: pd.DataFrame) -> bool:
        if not self.perfil.ranking_referencia or not self.perfil.municipios_referencia:
            return False
        comparables = 0
        coincidentes = 0
        for _, row in agrupado.iterrows():
            seccion = int(row["SECCION"])
            municipio_ref = self.perfil.municipios_referencia.get(seccion)
            if municipio_ref is None:
                continue
            comparables += 1
            if normalizar_clave(row["__MUNICIPIO"]) == normalizar_clave(municipio_ref):
                coincidentes += 1
        return comparables > 0 and (coincidentes / comparables) >= 0.8

    def _seleccionar_top_sin_duplicar_componentes(
        self,
        ranking: list[tuple[str, int | float, int]],
    ) -> list[tuple[str, int | float, int]]:
        seleccionados: list[tuple[str, int | float, int]] = []
        bloqueados: set[str] = set()
        componentes_por_indice = {
            idx: {normalizar_clave(componente) for componente in componentes}
            for idx, (_, componentes) in enumerate(self.perfil.ranking_grupos)
        }
        for etiqueta, valor, idx in ranking:
            clave = normalizar_clave(etiqueta)
            if clave in bloqueados:
                continue
            seleccionados.append((etiqueta, valor, idx))
            componentes = componentes_por_indice[idx]
            if len(componentes) > 1:
                bloqueados.update(componentes)
            if len(seleccionados) == 3:
                break
        return seleccionados

    def _serie_numerica_logica(self, df: pd.DataFrame, nombre: str, advertencias: list[str]) -> pd.Series:
        aliases = self.perfil.aliases_columnas.get(nombre, [nombre])
        columnas = self._resolver_columnas(df, aliases)
        if not columnas:
            if nombre in self.perfil.partidos_salida:
                advertencias.append(f"No se encontró la columna {nombre}; se llenó con 0.")
            return pd.Series([0] * len(df), index=df.index, dtype="float64")
        total = pd.Series([0] * len(df), index=df.index, dtype="float64")
        for columna in columnas:
            total = total + self._serie_numerica(df, [columna])
        return total

    def _sumar_componentes(self, df: pd.DataFrame, componentes: list[str], advertencias: list[str]) -> pd.Series:
        total = pd.Series([0] * len(df), index=df.index, dtype="float64")
        for componente in componentes:
            total = total + self._serie_numerica_logica(df, componente, advertencias)
        return total

    def _serie_texto(self, df: pd.DataFrame, aliases: list[str]) -> pd.Series:
        columna = self._resolver_columna(df, aliases)
        if columna is None:
            return pd.Series([""] * len(df), index=df.index, dtype="object")
        return df[columna].fillna("").astype(str).str.strip()

    def _serie_numerica(self, df: pd.DataFrame, aliases: list[str]) -> pd.Series:
        columna = self._resolver_columna(df, aliases)
        if columna is None:
            return pd.Series([0] * len(df), index=df.index, dtype="float64")
        serie = df[columna]
        if serie.dtype == object:
            serie = serie.astype(str).str.strip().str.replace(",", "", regex=False)
            serie = serie.replace({"": "0", " ": "0", "nan": "0", "None": "0"})
        return pd.to_numeric(serie, errors="coerce").fillna(0)

    @staticmethod
    def _resolver_columna(df: pd.DataFrame, aliases: list[str]) -> str | None:
        indice = {normalizar_clave(col): col for col in df.columns}
        for alias in aliases:
            columna = indice.get(normalizar_clave(alias))
            if columna is not None:
                return columna
        return None

    @staticmethod
    def _resolver_columnas(df: pd.DataFrame, aliases: list[str]) -> list[str]:
        indice = {normalizar_clave(col): col for col in df.columns}
        columnas: list[str] = []
        vistas: set[str] = set()
        for alias in aliases:
            columna = indice.get(normalizar_clave(alias))
            if columna is not None and columna not in vistas:
                columnas.append(columna)
                vistas.add(columna)
        return columnas

    def _columna_salida(self, opciones: list[str]) -> str:
        indice = {normalizar_clave(col): col for col in self.perfil.encabezados_visibles}
        for opcion in opciones:
            columna = indice.get(normalizar_clave(opcion))
            if columna is not None:
                return columna
        raise ValueError(f"El formato de referencia no contiene ninguna de estas columnas: {', '.join(opciones)}")

    @staticmethod
    def _primero_no_vacio(valores: pd.Series) -> str:
        for valor in valores:
            texto = "" if pd.isna(valor) else str(valor).strip()
            if texto:
                return texto
        return ""

    @staticmethod
    def _normalizar_numero(valor: object) -> int | float:
        numero = float(valor)
        return int(numero) if numero.is_integer() else numero
