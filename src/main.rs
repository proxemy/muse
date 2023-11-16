use gtk::prelude::*;
use gtk::{glib, Application, ApplicationWindow};


const APP_ID: &str = env!("CARGO_PKG_NAME");

fn main() -> glib::ExitCode
{
    let app = Application::builder().application_id(APP_ID).build();

    app.connect_activate(build_ui);

    app.run()
}


fn build_ui(app: &Application)
{
    let window = ApplicationWindow::builder()
        .application(app)
        .title(env!("CARGO_PKG_NAME"))
        .build();

    window.present();
}
