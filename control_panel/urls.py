# control_panel/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('dashboard/kill-switch/', views.global_kill_switch_api, name='global_kill_switch_api'),
    path('training/', views.training_view, name='training'),
    path('training/start/', views.start_training_job_view, name='start_training_job'),

    path('training/job/<int:job_id>/stop/', views.stop_job_view, name='stop_job'),
    path('training/meta/start/', views.start_meta_job_view, name='start_meta_job'),
    path('training/job/meta/stop/<int:job_id>/', views.stop_meta_job_view, name='stop_meta_job'),
    
    path('paper-trading/', views.papertrading_view, name='papertrading'),
    path('paper-trading/model/<int:trader_id>/', views.model_detail_view, name='model_detail'),
    path('paper-trading/start/', views.start_trader_view, name='start_trader'),
    path('paper-trading/stop/', views.stop_trader_view, name='stop_trader'),
    path('paper-trading/<int:trader_id>/stop/', views.stop_trader_view, name='stop_trader_instance'),
    path('paper-trading/<int:trader_id>/pause/', views.pause_trader_view, name='pause_trader'),
    path('paper-trading/<int:trader_id>/edit/', views.edit_trader_view, name='edit_trader_api'),
    path('paper-trading/<int:trader_id>/resume/', views.resume_trader_view, name='resume_trader'),
    path('paper-trading/<int:trader_id>/restart/', views.restart_trader_view, name='restart_trader'),
    path('paper-trading/start-all/', views.start_all_traders_view, name='start_all_traders'),
    path('paper-trading/stop-all/', views.stop_all_traders_view, name='stop_all_traders'),
    path('paper-trading/<int:trader_id>/delete/', views.delete_trader_api, name='delete_trader_api'),
    path('api/logs/trader/<int:trader_id>/', views.get_trader_log_api, name='api_trader_log'),
    
    path('evaluation/', views.evaluation_view, name='evaluation'),
    path('evaluation-lab/', views.evaluation_lab_view, name='evaluation_lab'),
    path('evaluation/report/<int:job_id>/', views.evaluation_report_view, name='evaluation_report'),
    
    path('models/', views.models_hub_view, name='models_hub'),
    path('api/models/delete/<str:file_name>/', views.delete_model_api, name='delete_model_api'),
    path('api/models/ready/<str:file_name>/', views.toggle_model_ready_api, name='toggle_model_ready_api'),
    
    path('guide/', views.onboarding_view, name='guide'),
    path('changelog/', views.changelog_view, name='changelog'),
    path('updates/', views.system_updates_view, name='system_updates'),
    
    # API Endpoints
    path('api/job-status/', views.job_status_api, name='job_status_api'),
    path('api/job-logs/<str:job_type>/<int:job_id>/', views.job_logs_api, name='job_logs_api'),
    path('api/trader-logs/<int:trader_id>/', views.trader_logs_api, name='trader_logs_api'),
    path('api/trader-status/', views.trader_status_api, name='trader_status_api'),
    path('api/trader-trades/<int:trader_id>/', views.trader_trades_api, name='api_trader_trades'),
    path('api/trader-activity/', views.trader_activity_api, name='trader_activity_api'),

    path('intelligence-vault/', views.reports_hub_view, name='reports_hub'),
    path('intelligence-vault/download/<int:report_id>/', views.download_report_pdf_view, name='download_report_pdf'),
    path('intelligence-vault/view/<int:report_id>/', views.view_report_view, name='view_report'),
    path('intelligence-vault/markdown/<int:report_id>/', views.download_report_markdown_view, name='download_report_markdown'),
    path('api/reports/<int:report_id>/content/', views.report_content_api, name='report_content_api'),

    path('trader-report/', views.trader_report_view, name='trader_report'),
    path('trader-report/reset/', views.reset_trader_report_view, name='reset_trader_report'),

    path('real-trading/', views.realtrading_view, name='realtrading'),
    path('real-trading/start/', views.start_real_trader_view, name='start_real_trader'),
    path('settings/', views.settings_view, name='settings'),
    path('settings/broker/add/', views.add_broker_account_api, name='add_broker_account_api'),
    path('settings/broker/delete/<int:account_id>/', views.delete_broker_account_api, name='delete_broker_account_api'),
    path('api/test-email/', views.test_email_api, name='test_email_api'),
    path('api/test-rewriter/', views.test_rewriter_api, name='test_rewriter_api'),
    path('api/test-ai-key/', views.test_ai_key_api, name='test_ai_key_api'),
    path('api/evolve/stream/', views.evolve_stream, name='evolve_stream'),
    path('api/alerts/<int:alert_id>/dismiss/', views.dismiss_alert_api, name='dismiss_alert_api'),
    path('api/settings/toggle/<str:setting_name>/', views.toggle_setting_api, name='toggle_setting_api'),
    path('api/mutate/', views.trigger_mutation_api, name='trigger_mutation_api'),
    path('api/mutation-logs/', views.mutation_logs_api, name='mutation_logs_api'),
    path('api/system-logs/', views.system_logs_api, name='system_logs_api'),
    path('api/system-telemetry/', views.system_telemetry_api, name='system_telemetry_api'),
    path('api/system/check-updates/', views.check_updates_api, name='check_updates_api'),
    path('api/system/update/stream/', views.system_update_stream, name='system_update_stream'),
    path('api/security/status/', views.security_status_api, name='security_status_api'),
    path('api/security/save/', views.save_security_settings_api, name='save_security_settings_api'),
    path('api/lock/', views.lockscreen_api, name='lockscreen_api'),

    # Neural Evolution Engine
    path('evolution-chamber/', views.evolution_hub_view, name='evolution_hub'),
    path('evolution-chamber/variant/<int:variant_id>/', views.variant_details_view, name='variant_details'),
    path('api/evolution/variants/<int:variant_id>/metrics/', views.variant_metrics_api, name='variant_metrics_api'),
    path('api/evolution/variants/', views.evolution_variants_api, name='evolution_variants_api'),
    path('api/evolution/variants/<int:variant_id>/logs/', views.variant_logs_api, name='variant_logs_api'),
    path('api/evolution/promote/<int:variant_id>/', views.evolution_promote_api, name='evolution_promote_api'),
    path('api/evolution/reject/<int:variant_id>/', views.evolution_reject_api, name='evolution_reject_api'),
    path('evolution-chamber/rejection-report/<int:alert_id>/', views.rejection_report_view, name='rejection_report'),
    path('api/evolution/delete/<int:variant_id>/', views.evolution_delete_api, name='evolution_delete_api'),
    path('api/evolution/restart/<int:variant_id>/', views.evolution_restart_api, name='evolution_restart_api'),
    path('api/evolution/evaluate/', views.evolution_evaluate_api, name='evolution_evaluate_api'),

    # Neural Cortex Dashboard & Weight Editor
    path('neural-cortex/', views.neural_cortex_view, name='neural_cortex'),
    path('api/neural/weights/', views.neural_weights_api, name='neural_weights_api'),
    path('api/neural/learning-log/', views.neural_learning_log_api, name='neural_learning_log_api'),
    path('api/neural/weights/edit/', views.neural_weight_edit_api, name='neural_weight_edit_api'),
    path('neural/download-report/', views.download_report_view, name='neural_download_report'),
    
    # Neural Decision Flow Analyzer
    path('neural/decision-flow/', views.decision_flow_view, name='neural_decision_flow'),
    path('api/neural/simulate-decision/', views.simulate_decision_api, name='simulate_decision_api'),
    path('api/neural/past-decisions/', views.past_decisions_api, name='past_decisions_api'),
    path('api/neural/record-decision/', views.record_decision_api, name='record_decision_api'),

    # Relax Lounge (Gaming Hub)
    path('relax/', views.relax_view, name='relax_view'),
    path('relax/browse/', views.relax_browse_dir, name='relax_browse_dir'),
    path('relax/serve-local-file/', views.serve_local_file, name='serve_local_file'),
    path('relax/game/sync-steam/', views.relax_sync_steam_playtimes, name='relax_sync_steam_playtimes'),
    path('relax/game/add/', views.relax_add_game, name='relax_add_game'),
    path('relax/game/edit/<int:game_id>/', views.relax_edit_game, name='relax_edit_game'),
    path('relax/game/delete/<int:game_id>/', views.relax_delete_game, name='relax_delete_game'),
    path('relax/guide/add/', views.relax_add_guide, name='relax_add_guide'),
    path('relax/guide/delete/<int:guide_id>/', views.relax_delete_guide, name='relax_delete_guide'),
    path('relax/video/add/', views.relax_add_video, name='relax_add_video'),
    path('relax/game/launch/<int:game_id>/', views.relax_launch_game, name='relax_launch_game'),
    path('relax/game/search-artwork/', views.relax_search_artwork, name='relax_search_artwork'),
    path('relax/game/detect-path/', views.relax_detect_game_path, name='relax_detect_game_path'),
    path('relax/analytics/', views.relax_analytics_view, name='relax_analytics'),
    path('relax/game/analytics/<int:game_id>/', views.relax_game_detail_analytics_view, name='relax_game_detail_analytics'),
    path('relax/watchlist/', views.relax_watchlist_view, name='relax_watchlist'),
    path('relax/watchlist/add-upcoming/', views.relax_add_watchlist_game, name='relax_add_watchlist_game'),
    path('relax/watchlist/delete-upcoming/<int:game_id>/', views.relax_delete_watchlist_game, name='relax_delete_watchlist_game'),
    path('relax/watchlist/add-budget/', views.relax_add_budget_game, name='relax_add_budget_game'),
    path('relax/watchlist/delete-budget/<int:game_id>/', views.relax_delete_budget_game, name='relax_delete_budget_game'),
    path('relax/immersion/', views.relax_immersion_view, name='relax_immersion'),
    path('relax/api/process-heartbeat/', views.relax_api_process_heartbeat, name='relax_api_process_heartbeat'),
    path('relax/api/immersion-status/', views.relax_api_immersion_status, name='relax_api_immersion_status'),
    path('relax/api/start-timer/', views.relax_api_start_timer, name='relax_api_start_timer'),
    path('relax/api/stop-session/', views.relax_api_stop_session, name='relax_api_stop_session'),
]


