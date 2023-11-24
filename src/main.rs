use gtk::prelude::*;
use gtk::{glib, Application, ApplicationWindow, Button};

use const_format::formatcp;


const APP_ID: &str = formatcp!("org.gtk_rs.{}", env!("CARGO_PKG_NAME"));

fn main() -> glib::ExitCode
{
    let app = Application::builder().application_id(APP_ID).build();

    app.connect_activate(build_ui);

    app.run()
}


fn build_ui(app: &Application)
{
    let button1 = Button::builder()
        .label("Process")
        .margin_top(12)
        .margin_bottom(12)
        .margin_start(12)
        .margin_end(12)
        .build();

    let button2 = Button::builder()
        .label("Process")
        .margin_top(12)
        .margin_bottom(12)
        .margin_start(12)
        .margin_end(12)
        .build();

    button1.connect_clicked(|button| {
        button.set_label("new label");
    });
    button1.connect_clicked(|button| {
        button.set_label("next label play");
    });

    let window = ApplicationWindow::builder()
        .application(app)
        .title(env!("CARGO_PKG_NAME"))
        .child(&button1)
        .child(&button2)
        .build();

    window.present();
}
