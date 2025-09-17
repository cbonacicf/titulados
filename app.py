#!/usr/bin/env python
# coding: utf-8

import polars as pl
import pickle

from io import BytesIO
import base64

import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# import dash
from dash import dcc
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Input, Output, State, DashProxy, MultiplexerTransform, html
from dash.exceptions import PreventUpdate

with open('./data/categorias_titul.pkl', 'rb') as f:
    categorias = pickle.load(f)

for k, v in categorias.items():
    exec(f'{k} = {v}')

# temporal
del map_tipo['U. CRUCH']
del map_tipo['U. Privadas']

del map_nivel['Carreras Técnicas']
del map_nivel['Carreras Profesionales']
del map_nivel['Posgrado']

titul = pl.scan_parquet('./data/titulados.parquet')

# ### Colores
with open('./data/colores_titul.pkl', 'rb') as f:
    colores = pickle.load(f)

# ### Funciones

inv = lambda dic: {v: k for k, v in dic.items()}

def base_datos(criterio, variable):
    mapa = eval(f'inv(map_{variable})')
    tipo = pl.Enum(list(mapa.values()))
    titul_loc = titul
    if criterio:
        titul_loc = titul.filter(**criterio)
    return (
        titul_loc
        .collect()
        .pivot(index=variable, on='ano', values='titulados', aggregate_function='sum')
        .with_columns(
            pl.col(variable).replace_strict(mapa, return_dtype=tipo)
        )
        .sort(variable)
    )


def base_grafico(base):
    variable = base.columns[0]
    return (
        base
        .cast({variable: pl.Utf8})
        .transpose(include_header=True, header_name='ano', column_names=variable)
        .cast({'ano': pl.UInt16})
    )


criterio = dict(nivel=3)   # default: pregrado

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
    ('Nivel', 'nivel', map_nivel, 3),
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

encabezado_tabla = dbc.Col([
    dbc.Row(html.H3('Tabla', style={'textAlign': 'center', 'marginTop': -10, 'marginBottom': 20}))
], width=9)

variables = [tupla[1] for tupla in tuplas]
map_nombres = dict((tupla[1], tupla[0]) for tupla in tuplas)


def crea_criterio(tipo, genero, nivel, region, area, stem):
    dic = dict(zip(variables, [tipo, genero, nivel, region, area, stem]))
    dic_retorna = {}
    for item in dic:
        if dic[item]:
            dic_retorna[item] = dic[item]
    return dic_retorna

# fmto = {'function': "d3.format('(,.0f')(params.value)"}
fmto = {"function": "d3.format(',.0f')(params.value).replace(/,/g, '.')"}

def crea_column_defs(variable):
    return [
        {'field': variable, 'headerName': map_nombres[variable], 'width': 250, 'type': 'leftAligned', 'pinned': 'left'},
        {'field': '2010', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
        {'field': '2011', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
        {'field': '2012', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
        {'field': '2013', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
        {'field': '2014', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
        {'field': '2015', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
        {'field': '2016', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
        {'field': '2017', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
        {'field': '2018', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
        {'field': '2019', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
        {'field': '2020', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
        {'field': '2021', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
        {'field': '2022', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
        {'field': '2023', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
        {'field': '2024', 'width': 100, 'type': 'numericColumn', 'valueFormatter': fmto},
    ]

# tabla de datos
getRowStyle = {
    'styleConditions': [{
        'condition': 'params.rowIndex % 2 === 0',
        'style': {'backgroundColor': 'rgb(47, 164, 231, 0.1)'},
    }],
}


def tabla_datos(criterio, variable):
    row_data = base_datos(criterio, variable).to_dicts()

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


# botón que exporta selección a excel
btn_exp_datos = dbc.Row([
    html.Button('Exportar a Excel',
        id='exporta-datos',
        className='btn btn-outline-primary',
        style={'width': '15%', 'marginRight': 10, 'marginTop': 10, 'padding': '6px 15px'},
    ),
    dcc.Download(id='exporta-datos-archivo'),
], justify='end',)


def exporta_datos(datos):
    output = BytesIO()
    (
        pl.DataFrame(datos)
        .select([pl.col(pl.String)]+[str(i) for i in range(2010, 2025)])
        .write_excel(workbook=output, autofilter=False)
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
    btn_exp_datos,
    footer,

    dcc.Store(id='datos-exporta', data=base_datos(criterio, 'tipo').to_dicts()),
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

    Input('restablece', 'n_clicks'),
    prevent_initial_call=True,
)
def restablece_seleccion(click):
    if click == 0:
        raise PreventUpdate
    else:
        defecto = [0, 0, 3, 0, 0, 0]
        crt_local = crea_criterio(*defecto)
        base_local = base_datos(crt_local, 'tipo').to_dicts()
        return base_local, crea_column_defs('tipo'), *defecto, 'tipo', 0, base_local


# modifica selección
@app.callback(
    Output('tabla-datos', 'rowData'),
    Output('tabla-datos', 'columnDefs'),
    Output(component_id='imagen-grafico', component_property='src'),
    Output('datos-exporta', 'data'),

    Input('drop-tipo', 'value'),
    Input('drop-genero', 'value'),
    Input('drop-nivel', 'value'),
    Input('drop-region', 'value'),
    Input('drop-area', 'value'),
    Input('drop-stem', 'value'),
    Input('boton-radio', 'value'),
    Input('boton-radio2', 'value'),
)
def modifica_seleccion(tipo, genero, nivel, region, area, stem, var_local, tipo_graf):
    crt_local = crea_criterio(tipo, genero, nivel, region, area, stem)
    base_local = base_datos(crt_local, var_local)
    return base_local.to_dicts(), crea_column_defs(var_local), crea_figura(base_grafico(base_local), tipo_graf), base_local.to_dicts()


# exporta datos a excel
@app.callback(
    Output('exporta-datos-archivo', 'data'),
    Input('exporta-datos', 'n_clicks'),
    State('datos-exporta', 'data'),
    prevent_initial_call=True,
)
def exporta_datos_excel(_, datos):
    df = exporta_datos(datos)
    return dcc.send_bytes(df, 'datos_titulados.xlsx')


# ejecución de la aplicación
if __name__ == '__main__':
    app.run()
