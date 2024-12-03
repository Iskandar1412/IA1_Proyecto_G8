import { NavLink, useLocation } from "react-router-dom";

export const SidebarMenuItem = ({ path, icon, title, subTitle }) => {
  const location = useLocation();
  const isPathCurrent = path === location.pathname;

  return (
    <NavLink
      to={path}
      className={`w-full px-2 inline-flex space-x-2 items-center border-b border-slate-700 py-3 ${
        isPathCurrent ? "bg-blue-800" : ""
      } hover:bg-white/5 transition ease-linear duration-150`}
    >
      <div>{icon}</div>
      <div className="flex flex-col">
        <span className="text-lg font-bold leading-5 text-white">{title}</span>
        <span className="text-sm text-white/50 hidden md:block">
          {subTitle}
        </span>
      </div>
    </NavLink>
  );
};
