import React from 'react'
import {  IoChatbox, IoLogoReact } from "react-icons/io5";
import { SidebarMenuItem } from './SideBarMenuItem';


const menuItems = [
    {
      path: "/",
      icon: <IoChatbox size={40} />,
      title: "Chat",
      subTitle: "Nueva Funcionalidad.",
    },
   
  ];

export const SideBar = () => {
  return (
    <div
    id="menu"
    style={{ width: "400px" }}
    className="bg-gray-900 min-h-screen z-10 text-slate-300 w-64  left-0 "
  >
    <div id="logo" className="my-4 px-6">
      <h1 className="flex items-center text-lg md:text-2xl font-bold text-white">
        <IoLogoReact className="mr-2" />
        <span>IaChat</span>
        <span className="text-blue-500"> 🤖</span>.
      </h1>
      <p className="text-slate-500 text-sm">
        Chatea con el IaChat para resolver tus dudas.
      </p>
    </div>
    <div id="profile" className="px-6 py-10">
      <p className="text-slate-500">Bienvendio de vuelta,</p>
      <a href="#" className="inline-flex space-x-2 items-center">
        <span>
          <img
            className="rounded-full w-8 h-8"
            src={"https://images.unsplash.com/photo-1542909168-82c3e7fdca5c"}
            alt="User Avatar"
            width={50}
            height={50}
          />
        </span>
        <span className="text-sm md:text-base font-bold">Alexis López</span>
      </a>
    </div>
    <div id="nav" className="w-full px-6">
      {menuItems.map((item, index) => (
        <SidebarMenuItem
          key={index}
          path={item.path}
          icon={item.icon}
          title={item.title}
          subTitle={item.subTitle}
        />
      ))}
    </div>
  </div>
  )
}
