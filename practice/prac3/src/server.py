import io

import numpy as np
import plotly
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired, NumberRange, ValidationError
from wtforms import SelectField, IntegerField, SubmitField, FileField, DecimalField
from model import Model
import pandas as pd
import ast

import plotly.graph_objs as go
import json

app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'my_secret_key'
Bootstrap(app)

model = None
preds = None


@app.errorhandler(404)
def handle_404(e):
    return redirect(url_for('get_text_score'))


class SelectForm(FlaskForm):
    model = SelectField('Модель', choices=[(0, 'Случайный лес'), (1, 'Градиентный бустинг')], default=0)
    n_estimators = IntegerField('Число базовых моделей',
                                validators=[DataRequired(), NumberRange(1)], default=100)
    depth = IntegerField('Максимальная глубина одного дерева',
                         validators=[DataRequired(), NumberRange(1)], default=7)

    feature_subsample_size = DecimalField('Доля случайных признаков, на которых будет обучена модель',
                                          default=1., validators=[DataRequired(), NumberRange(0., 1.)])
    learning_rate = DecimalField('Шаг градиентного спуска *используется только при градиентном бустинге',
                                 default=0.1, validators=[DataRequired(), NumberRange(0., 1.)])

    train_path = FileField('Обучающая выборка', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'Только .csv формат.')
    ])

    valid_path = FileField('Валидационная выборка', validators=[
        FileAllowed(['csv'], 'Только .csv формат.')
    ])

    load_data_submit = SubmitField('Обучить модель')


class LoadDataForPredict(FlaskForm):
    test_path = FileField('Тестовая выборка', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'Только .csv формат.')
    ])
    load_data_submit = SubmitField('Предсказать')


class ExitButton(FlaskForm):
    exit = SubmitField('Вернуться на начальную страницу')


def create_plot(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(history['train_loss'])), y=history['train_loss'], name='train loss'))
    fig.add_trace(go.Scatter(x=np.arange(len(history['valid_loss'])), y=history['valid_loss'], name='valid loss'))
    fig.update_layout(legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"),
                      title="Функция потерь от количества обученных базовых моделей",
                      xaxis_title="Число базовых моделей",
                      yaxis_title="Средняя квадратичная ошибка (RMSE)",
                      margin=dict(l=0, r=0, t=30, b=0))
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.route('/model', methods=['GET', 'POST'])
def model_info():
    n, m = request.args.get('n', default=None, type=int), request.args.get('m', default=None, type=int)
    global model
    if model is not None:
        model_info = model.get_info()
        model_info.extend([n, m])
    else:
        model_info = ['Модель еще не обучена', '-', '-', '-', '-', '-', '-']
    history_request = request.args.get('history', default=None)

    if history_request is not None:
        history = ast.literal_eval(history_request)
        time = str(round(history['time'], 2)) + 'c'
        graphJSON = create_plot(history)
    else:
        graphJSON = None
        time = '-'

    data_loader = LoadDataForPredict()
    try:
        if request.method == 'POST' and data_loader.validate_on_submit():
            lines = data_loader.test_path.data.stream.readlines()
            test = pd.read_csv(io.StringIO("".join([str(line, "UTF-8") for line in lines])), delimiter=",")
            if test.shape[1] != m:
                raise ValidationError('Количество столбцов тестовой выборки должно быть равно ' + str(m))
            global preds
            preds = pd.DataFrame(model.predict(test.to_numpy()), columns=['price']).astype(int)
            return redirect(url_for('end'))

        return render_template('model_info.html', model_info=model_info,
                               time=time, graphJSON=graphJSON, load_file=data_loader)
    except Exception as exc:
        app.logger.info('Exeption {0}'.format(exc))
        return render_template('model_info.html', model_info=model_info,
                               time=time, graphJSON=graphJSON, load_file=data_loader)

@app.route('/', methods=['GET', 'POST'])
def index():
    model_selection_form = SelectForm()
    try:
        if request.method == 'POST' and model_selection_form.validate_on_submit():
            params = [i for i in request.form.values()][1:-1]

            global model
            model = Model(params)

            lines = model_selection_form.train_path.data.stream.readlines()
            train = pd.read_csv(io.StringIO("".join([str(line, "UTF-8") for line in lines])), delimiter=",")
            lines = model_selection_form.valid_path.data.stream.readlines()
            if len(lines) > 0:
                valid = pd.read_csv(io.StringIO("".join([str(line, "UTF-8") for line in lines])), delimiter=",")
            else:
                valid = None
            try:
                y_train = train['price'].to_numpy(dtype='float32')
                X_train = train.drop('price', axis=1).to_numpy(dtype='float32')
                if valid is not None:
                    y_valid = valid['price'].to_numpy(dtype='float32')
                    X_valid = valid.drop('price', axis=1).to_numpy(dtype='float32')

                    if X_valid.shape[1] != X_train.shape[1]:
                        raise ValidationError
                else:
                    y_valid = None
                    X_valid = None

                history = model.fit(X_train, y_train, X_valid, y_valid)
                return redirect(url_for('model_info', history=history, n=X_train.shape[0],
                                        m=X_train.shape[1]))
            except Exception as exc:
                app.logger.info('Exeption {0}'.format(exc))

        return render_template('from_form.html',
                               model_selection_form=model_selection_form)
    except Exception as exc:
        app.logger.info('Exeption {0}'.format(exc))
        return render_template('from_form.html',
                               model_selection_form=model_selection_form)


@app.route('/predictions', methods=['GET', 'POST'])
def end():
    exit_btn = ExitButton()
    if request.method == 'POST' and exit_btn.validate_on_submit():
        return redirect(url_for('index'))
    return render_template('exit.html', preds=preds,
                           exit=exit_btn)

