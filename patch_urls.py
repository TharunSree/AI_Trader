with open('control_panel/urls.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = "    path('settings/toggle/<str:setting_name>/', views.toggle_setting_api, name='toggle_setting_api'),"
replacement = '''    path('settings/toggle/<str:setting_name>/', views.toggle_setting_api, name='toggle_setting_api'),
    path('settings/broker/add/', views.add_broker_account_api, name='add_broker_account_api'),
    path('settings/broker/delete/<int:account_id>/', views.delete_broker_account_api, name='delete_broker_account_api'),'''

if "add_broker_account_api" not in text:
    text = text.replace(target, replacement)
    with open('control_panel/urls.py', 'w', encoding='utf-8') as f:
        f.write(text)
