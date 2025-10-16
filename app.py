#!/usr/bin/env python
# coding: utf-8

import polars as pl
import pickle
from collections import namedtuple

from io import BytesIO
import base64

import matplotlib as mpl
mpl.use('agg')

import xlsxwriter

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import dash
from dash import dcc
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Input, Output, State, DashProxy, MultiplexerTransform, html
from dash.exceptions import PreventUpdate

with open('./data/categorias_titul.pkl', 'rb') as f:
    categorias = pickle.load(f)

for k, v in categorias.items():
    exec(f'{k} = {v}')

titul = pl.scan_parquet('./data/titulados.parquet')

# ### Colores
with open('./data/colores_titul.pkl', 'rb') as f:
    colores = pickle.load(f)


# ### Clase

class Datos:
    K = [float('inf'), -float('inf')]

    def __init__(self, df):
        self.df = df
        self.col = df.columns[0]
        self.cols = df.columns[1:]
        self.pct = self.porcentaje()
        self.var = self.variacion()
        self.dif = self.diferencia()

    def porcentaje(self):
        return (
            self.df
            .cast({self.col: pl.Utf8})
            .filter(pl.col(self.col) != 'Total')
            .with_columns(
                [(pl.col(c)/pl.col(c).sum()).round(3) for c in self.cols]
            )
        )

    def variacion(self):
        return (
            self.df
            .with_columns(
                [(pl.col(act)/pl.col(ant)-1).round(3).replace(Datos.K, None) for act, ant in zip(self.cols[1:], self.cols[:-1])]
            )
            .drop(self.cols[0])
        )

    def var_acc(self, ano=2020):
        return (
            self.df
            .with_columns(
                [(pl.col(c)/pl.col(str(ano))-1).round(3).replace(Datos.K, None) for c in self.cols]
            )
        )

    def diferencia(self):
        return (
            self.df
            .with_columns(
                [(pl.col(act)-pl.col(ant)) for act, ant in zip(self.cols[1:], self.cols[:-1])]
            )
            .drop(self.cols[0])
        )

    def dif_acc(self, ano=2020):
        return (
            self.df
            .with_columns(
                [(pl.col(c)-pl.col(str(ano))) for c in self.cols]
            )
        )


# ### Funciones

inv = lambda dic: {v: k for k, v in dic.items()}

def total(base):
    variable = base.columns[0]
    return pl.DataFrame({variable: 'Total'}).join(base.select(pl.exclude(variable)).sum(), how='cross')


def base_datos(criterio, variable):
    mapa = inv(eval(f'map_{variable}'))
    tipo = pl.Enum(list(mapa.values()) + ['Total'])
    df = (
        titul
        .filter(**criterio)
        .collect()
        .pivot(index=variable, on='ano', values='titulados', aggregate_function='sum')
        .with_columns(
            pl.col(variable).replace_strict(mapa, return_dtype=pl.Utf8).alias(variable)
        )
    )
    if len(df) > 1:
        return df.cast({variable: tipo}).sort(variable), pl.concat([df, total(df)]).cast({variable: tipo}).sort(variable)
    else:
        return df, df

def base_grafico(base):
    variable = base.columns[0]
    return (
        base
        .cast({variable: pl.Utf8})
        .transpose(include_header=True, header_name='ano', column_names=variable)
        .cast({'ano': pl.UInt16})
    )


criterio = dict(nivel=1)   # default: pregrado

# ### Gráficos

def crea_figura(datos, tipo):
    output = BytesIO()

    fig, ax = plt.subplots(figsize=(11, 6))

    if tipo == 0:
        for item in datos.columns[1:]:
            ax.plot(datos.get_column('ano'), datos.get_column(item)/1000, label=item, color=colores[item], lw=2)
    elif tipo == 1:
        datos_loc = datos.fill_null(strategy='zero')
        ax.stackplot(
            datos_loc.get_column('ano').to_list(), *[(datos_loc.get_column(item)/1000).to_list() for item in datos.columns[1:]],
            labels=datos.columns[1:],
            colors=[colores[i] for i in datos.columns[1:]],
            alpha=0.8,
        )

    ax.set_xticks(datos.get_column('ano'))
    ax.set_ylabel('Cantidad (en miles)', fontsize=10)
    ax.spines[['right', 'top']].set_visible(False)
    ax.margins(x=0.01)
    ax.legend(ncols=3, frameon=False, bbox_to_anchor=(0.5, -0.075), loc='upper center', fontsize=10)

    fig.savefig(output, format='png', bbox_inches='tight')
    fig_data = base64.b64encode(output.getbuffer()).decode('ascii')
    plt.close()
    return f'data:image/png;base64,{fig_data}'


# ### Formas

# color azul de tab, botones, footer, etc.
color = '#2FA4E7'


# #### Encabezado

# encabezado
encabezado = html.Div(
    dbc.Row([
        dbc.Col(
            html.Img(src='./assets/cup-logo3.png', style={'width': '100%', 'height': '100%'}),
            width=2,
        ),
        dbc.Col(
            html.H1(['Información de Titulados'], style={'textAlign': 'center'}),
            width=7
        ),
    ], align='center'),
    style={'marginTop': 15}
)

# #### Dropdown

def crea_opciones(dic):
    return [{'label': k, 'value': v} for k, v in dic.items()]

def drop_down(identidad, dic, ini):
    return dcc.Dropdown(
        id=f'drop-{identidad}',
        options=crea_opciones(dic),
        value=ini,
        style={'fontSize': '14px'},
        clearable=False,
    )

sty_encabezado = {'fontSize': '16px', 'marginTop': 15, 'marginBottom': 0}

def dropdown_block(encabezado, variable, mapa, inicio):
    return dbc.Row(
        dbc.Col([
            html.H6(encabezado, style=sty_encabezado),
            drop_down(variable, mapa, inicio),
        ]), justify='center'
    )

tuplas = [
    ('Tipo de institución', 'tipo', map_tipo, 0),
    ('Género', 'genero', map_genero, 0),
    ('Nivel', 'nivel', map_nivel, 1),
    ('Región', 'region', map_region, 0),
    ('Área del conocimiento', 'area', map_area, 0),
    ('Carreras STEM', 'stem', map_stem, 0),
]

def desplegable(tuplas):
    return dbc.Col(
        [dropdown_block(*tupla) for tupla in tuplas]
    )


# #### Nucleo

boton = html.Button('Restablecer selección',
    id='restablece',
    style={'width': '200px'},
    className='btn btn-outline-primary'
)

op_btn_radio = crea_opciones(dict((tupla[0], tupla[1]) for tupla in tuplas))

boton_radio = dcc.RadioItems(
    id = 'boton-radio',
    options = op_btn_radio,
    value = 'tipo',
    style = {'textAlign': 'center'},
    labelStyle = {'display': 'inline-block', 'fontSize': '14px', 'fontWeight': 'normal'},
    inputStyle = {'marginRight': '5px', 'marginLeft': '20px'},
),

op_btn_radio2 = crea_opciones({'Líneas': 0, 'Áreas apiladas': 1})

boton_radio2 = dcc.RadioItems(
    id = 'boton-radio2',
    options = op_btn_radio2,
    value = 0,
    style = {'textAlign': 'center'},
    labelStyle = {'display': 'inline-block', 'fontSize': '14px', 'fontWeight': 'normal'},
    inputStyle = {'marginRight': '5px', 'marginLeft': '20px'},
),

op_btn_radio_ano = crea_opciones({str(k): k for k in list(range(2010, 2026))})

boton_radio_anos = html.Div([
    html.P("Seleccione el año de referencia:", style={'margin-left': '20px', 'margin-bottom': '0'}),
    dcc.RadioItems(
        id='btn-radio-ano',
        options = op_btn_radio_ano,
        inline=True,
        labelStyle = {'display': 'inline-block', 'fontSize': '14px', 'fontWeight': 'normal'},
        inputStyle = {'marginRight': '4px', 'marginLeft': '16px'},
    )
], id='div-btn-radio-ano', style={'display': 'flex', 'align-items': 'center', 'marginBottom': '15px'}, hidden=True)

def nucleo():
    return html.Div(
        dbc.Row([
            # Gráfico
            dbc.Col([
                dbc.Row(html.H3('Gráfico', style={'textAlign': 'center', 'marginTop': -10})),
                dbc.Row(boton_radio2),
                dbc.Row(boton_radio),
                html.Img(id='imagen-grafico')
            ], width=9),
            # Controles
            dbc.Col([
                dbc.Row(html.H5('Selección de la muestra', style={'textAlign': 'center'})),
                desplegable(tuplas),
                html.Br(),
                dbc.Row(boton, justify='center'),
            ], width=3),
        ])
    )


# #### Tabla

locale_es_CL = """d3.formatLocale({
  "decimal": ",",
  "thousands": ".",
  "grouping": [3],
  "currency": ["$", ""]
})"""

encabezado_tabla = dbc.Col([
    dbc.Row(html.H3('Tabla', style={'textAlign': 'center', 'marginTop': -10, 'marginBottom': 20}))
], width=9)

orden = ['tipo', 'genero', 'nivel', 'region', 'area', 'stem'] + [str(x) for x in range(2010, 2025)]
Crt = namedtuple('Crt', ['tipo', 'genero', 'nivel', 'region', 'area', 'stem'])

variables = [tupla[1] for tupla in tuplas]
map_nombres = dict((tupla[1], tupla[0]) for tupla in tuplas)

def crea_criterio(tipo, genero, nivel, region, area, stem):
    dic = dict(zip(variables, [tipo, genero, nivel, region, area, stem]))
    dic_retorna = {}
    for item in dic:
        if dic[item]:
            dic_retorna[item] = dic[item]
    return dic_retorna

fmto = {"function": f"{locale_es_CL}.format(',.0f')(params.value)"}

lista_fmto = [',.0f', ',.1%']
fn_fmto = lambda n: {"function": f"{locale_es_CL}.format('{lista_fmto[n]}')(params.value)"}

def crea_column_defs(variable):
    return [
        {'field': variable, 'headerName': map_nombres[variable], 'width': 250, 'type': 'leftAligned', 'pinned': 'left'},
        {'field': '2010', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
        {'field': '2011', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
        {'field': '2012', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
        {'field': '2013', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
        {'field': '2014', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
        {'field': '2015', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
        {'field': '2016', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
        {'field': '2017', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
        {'field': '2018', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
        {'field': '2019', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
        {'field': '2020', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
        {'field': '2021', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
        {'field': '2022', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
        {'field': '2023', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
        {'field': '2024', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(0)},
    ]

# tabla de datos
getRowStyle = {
    'styleConditions': [{
        'condition': 'params.rowIndex % 2 === 0',
        'style': {'backgroundColor': 'rgb(47, 164, 231, 0.1)'},
    }],
}


def tabla_datos(criterio, variable):
    row_data = base_datos(criterio, variable)[1].to_dicts()

    return dag.AgGrid(
        id='tabla-datos',
        rowData=row_data,
        defaultColDef={'resizable': True},
        columnDefs=crea_column_defs(variable),
        dashGridOptions = {
            'headerHeight': 40,
            'rowHeight': 35,
            'domLayout': 'autoHeight',
            'rowSelection': 'single',
        },
        getRowStyle=getRowStyle,
        style={'width': 1296}
    )

encabezado_tabla2 = dbc.Col([
    dbc.Row([
        html.H3('Transformación de datos:', style={'textAlign': 'left', 'marginLeft': 20, 'marginTop': -10, 'marginBottom': 0}),
    ]),
], width=3)

selector_tabla = dbc.Col(
    dcc.Dropdown(id='drop-trans',
       options=[
           {'label': 'Distribución porcentual anual', 'value': 1},
           {'label': 'Variación con respecto al año anterior', 'value': 2},
           {'label': 'Variación con respecto a un año determinado', 'value': 3},
           {'label': 'Diferencia con respecto al año anterior', 'value': 4},
           {'label': 'Diferencia con respecto a un año determinado', 'value': 5},
       ],
       value=1,
       clearable=False,
    ),
    width=4,
    style={'marginLeft': 20, 'marginBottom': 15}
)

def crea_column_defs2(variable, n, cols):
    return [
        {'field': variable, 'headerName': map_nombres[variable], 'width': 250, 'type': 'leftAligned', 'pinned': 'left'}
    ] + [{'field': f'{i}', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fn_fmto(n)} for i in cols]

def tabla_datos2(criterio, variable):
    row_data = base_datos(criterio, variable)[1]
    dt = Datos(row_data)

    return dag.AgGrid(
        id='tabla-datos2',
        rowData=dt.pct.to_dicts(),
        defaultColDef={'resizable': True},
        columnDefs=crea_column_defs2(variable, 1, dt.cols),
        dashGridOptions = {
            'headerHeight': 40,
            'rowHeight': 35,
            'domLayout': 'autoHeight',
            'rowSelection': 'single',
        },
        getRowStyle=getRowStyle,
        style={'width': 1296}
    )

# botón que exporta selección a excel
btn_exp_datos = dbc.Row([
    html.Button('Exportar a Excel',
        id='exporta-datos',
        className='btn btn-outline-primary',
        style={'width': '15%', 'marginRight': 10, 'marginTop': 10, 'padding': '6px 15px'},
    ),
    dcc.Download(id='exporta-datos-archivo'),
], justify='end',)

import polars.selectors as cs

custom_formats = {
    cs.integer(): "#,##0;-#,##0",
    cs.float(): "#,##0.0%;-#,##0.0%",
}


def exporta_datos(datos, trans, param):
    output = BytesIO()

    df = (
        pl.DataFrame(datos)
        .select([pl.col(pl.String)]+[str(i) for i in range(2010, 2025)])
    )

    dt = Datos(df)
    pos = (2, 0)

    match trans:
        case 1:
            tabla = dt.pct
        case 2:
            tabla = dt.var
        case 3:
            tabla = dt.var_acc(param['3'])
        case 4:
            tabla = dt.dif
        case 5:
            tabla = dt.dif_acc(param['5'])

    with xlsxwriter.Workbook(output) as workbook:
        titulo = workbook.add_format({'font_size': 16})

        worksheet = workbook.add_worksheet('Datos')
        worksheet.set_column(1, 16, 10)
        worksheet.write(0, 0, 'Datos de Titulación', titulo)
        df.write_excel(workbook=workbook, worksheet='Datos', position=pos, autofilter=False)

        tabla.write_excel(
            workbook=workbook,
            worksheet='Datos',
            position=(pos[0]+len(df)+2, pos[1]),
            column_formats=custom_formats,
            autofilter=False,
            autofit=True,
        )

    return output.getvalue()


footer = html.Div(
    html.Footer(
        style={
            'display': 'flex',
            'background': color,
            'padding': '10px',
            'marginTop': 25,
        }
    )
)


# ### Aplicación

app = DashProxy(__name__, transforms=[MultiplexerTransform()], external_stylesheets=[dbc.themes.CERULEAN])

app.config.suppress_callback_exceptions = True

server = app.server

# layout de la aplicación
app.layout = dbc.Container([
    encabezado,
    html.Hr(),
    nucleo(),
    html.Hr(),
    encabezado_tabla,
    tabla_datos(criterio, 'tipo'),
    html.Hr(),
    encabezado_tabla2,
    selector_tabla,
    boton_radio_anos,
    tabla_datos2(criterio, 'tipo'),
    btn_exp_datos,
    footer,

    dcc.Store(id='datos-exporta', data=base_datos(criterio, 'tipo')[1].to_dicts()),
    dcc.Store(id='param', data={'3': 2020, '5': 2020}),
])

# callbacks
# reestablece selección inicial
@app.callback(
    Output('tabla-datos', 'rowData'),
    Output('tabla-datos', 'columnDefs'),
    Output('drop-tipo', 'value'),
    Output('drop-genero', 'value'),
    Output('drop-nivel', 'value'),
    Output('drop-region', 'value'),
    Output('drop-area', 'value'),
    Output('drop-stem', 'value'),
    Output('boton-radio', 'value'),
    Output('boton-radio2', 'value'),
    Output('datos-exporta', 'data'),

    Output('tabla-datos2', 'rowData'),
    Output('tabla-datos2', 'columnDefs'),
    Output('param', 'data'),
    Output('drop-trans', 'value'),
    Output('div-btn-radio-ano', 'hidden'),

    Input('restablece', 'n_clicks'),
    prevent_initial_call=True,
)
def restablece_seleccion(click):
    if click == 0:
        raise PreventUpdate
    else:
        defecto = [0, 0, 1, 0, 0, 0]
        crt_local = crea_criterio(*defecto)
        base_local = base_datos(crt_local, 'tipo')[1]
        dt = Datos(base_local)
        return base_local.to_dicts(), crea_column_defs('tipo'), *defecto, 'tipo', 0, base_local.to_dicts(), dt.pct.to_dicts(), crea_column_defs2(dt.col, 1, dt.cols), \
            {'3': 2020, '5': 2020}, 1, True


# modifica selección
@app.callback(
    Output('tabla-datos', 'rowData'),
    Output('tabla-datos', 'columnDefs'),
    Output(component_id='imagen-grafico', component_property='src'),
    Output('datos-exporta', 'data'),

    Output('tabla-datos2', 'rowData'),
    Output('tabla-datos2', 'columnDefs'),

    Input('drop-tipo', 'value'),
    Input('drop-genero', 'value'),
    Input('drop-nivel', 'value'),
    Input('drop-region', 'value'),
    Input('drop-area', 'value'),
    Input('drop-stem', 'value'),
    Input('boton-radio', 'value'),
    Input('boton-radio2', 'value'),

    State('drop-trans', 'value'),
    State('param', 'data'),
)
def modifica_seleccion(tipo, genero, nivel, region, area, stem, var_local, tipo_graf, trans, param):
    crt = crea_criterio(tipo, genero, nivel, region, area, stem)
    df, dft = base_datos(crt, var_local)
    dt = Datos(dft)
    match trans:
        case 1:
            tabla, columnas = dt.pct.to_dicts(), crea_column_defs2(dt.col, 1, dt.cols)
        case 2:
            tabla, columnas = dt.var.to_dicts(), crea_column_defs2(dt.col, 1, dt.cols[1:])
        case 3:
            tabla, columnas = dt.var_acc(param['3']).to_dicts(), crea_column_defs2(dt.col, 1, dt.cols)
        case 4:
            tabla, columnas = dt.dif.to_dicts(), crea_column_defs2(dt.col, 0, dt.cols[1:])
        case 5:
            tabla, columnas = dt.dif_acc(param['5']).to_dicts(), crea_column_defs2(dt.col, 0, dt.cols)
    return dft.to_dicts(), crea_column_defs(var_local), crea_figura(base_grafico(df), tipo_graf), dft.to_dicts(), tabla, columnas

# modifica selección de transformación
@app.callback(
    Output('tabla-datos2', 'rowData'),
    Output('tabla-datos2', 'columnDefs'),
    Input('drop-trans', 'value'),
    State('datos-exporta', 'data'),
    State('param', 'data'),
    prevent_initial_call=True,
)
def cambia_tabla_transformacion(trans, data, param):
    df = pl.DataFrame(data)
    df = df.select([col for col in orden if col in df.columns])
    dt = Datos(df)
    match trans:
        case 1:
            tabla, columnas = dt.pct.to_dicts(), crea_column_defs2(dt.col, 1, dt.cols)
        case 2:
            tabla, columnas = dt.var.to_dicts(), crea_column_defs2(dt.col, 1, dt.cols[1:])
        case 3:
            tabla, columnas = dt.var_acc(param['3']).to_dicts(), crea_column_defs2(dt.col, 1, dt.cols)
        case 4:
            tabla, columnas = dt.dif.to_dicts(), crea_column_defs2(dt.col, 0, dt.cols[1:])
        case 5:
            tabla, columnas = dt.dif_acc(param['5']).to_dicts(), crea_column_defs2(dt.col, 0, dt.cols)
    return tabla, columnas

# selecciona transformación
@app.callback(
    Output('div-btn-radio-ano', 'hidden'),
    Output('btn-radio-ano', 'value'),
    Input('drop-trans', 'value'),
    State('param', 'data'),
    prevent_initial_call=True,
)
def selecciona_transformacion(trans, param):
    dic = {1: True, 2: True, 3: False, 4: True, 5: False}
    return dic[trans], param.get(str(trans), dash.no_update)

# selecciona año de referencia
@app.callback(
    Output('tabla-datos2', 'rowData'),
    Output('tabla-datos2', 'columnDefs'),
    Output('param', 'data'),
    Input('btn-radio-ano', 'value'),
    State('drop-trans', 'value'),
    State('datos-exporta', 'data'),
    State('param', 'data'),
    prevent_initial_call=True,
)
def slecciona_ano(ano, trans, data, param):
    df = pl.DataFrame(data)
    df = df.select([col for col in orden if col in df.columns])
    dt = Datos(df)
    param[str(trans)] = ano
    match trans:
        case 3:
            tabla, columnas = dt.var_acc(param['3']).to_dicts(), crea_column_defs2(dt.col, 1, dt.cols)
        case 5:
            tabla, columnas = dt.dif_acc(param['5']).to_dicts(), crea_column_defs2(dt.col, 0, dt.cols)
    return tabla, columnas, param


# exporta datos a excel
@app.callback(
    Output('exporta-datos-archivo', 'data'),
    Input('exporta-datos', 'n_clicks'),
    State('datos-exporta', 'data'),
    State('drop-trans', 'value'),
    State('param', 'data'),
    prevent_initial_call=True,
)
def exporta_datos_excel(_, datos, trans, param):
    df = exporta_datos(datos, trans, param)
    return dcc.send_bytes(df, 'datos_titulados.xlsx')


# ejecución de la aplicación
if __name__ == '__main__':
    app.run(debug=True, port=8055)
