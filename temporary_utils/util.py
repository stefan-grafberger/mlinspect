import dash
import json

POS_DICT = {}


def util():
    trigger = dash.callback_context.triggered[0]
    trigger_prop = trigger['prop_id'].split('.')[-1]
    trigger_value = trigger['value']
    if trigger_value:
        point = trigger_value['points'][0]
        x = point['x']
        y = point['y']
        try:
            node = [node for node, pos in POS_DICT.items() if pos == (x, y)][0]
        except IndexError:
            print(f"[{trigger_prop}] Could not find node with pos {x} and {y}")
            return dash.no_update, dash.no_update, dash.no_update
        code_ref = node.code_reference
        return json.dumps(code_ref.__dict__), dash.no_update, dash.no_update
    else:
        return [], dash.no_update, dash.no_update

    return dash.no_update, dash.no_update, dash.no_update
